"""Two independent rendering layers for `UltrasoundProbeEnv`.

1. `render_matplotlib` -- the default, headless-safe renderer used by
   `main.py` and `env.render()`. Uses the "Agg" backend explicitly so it
   never requires a display. Draws: (a) a 3D probe-on-abdomen scatter with
   a target-plane indicator, (b) the current simulated ultrasound slice,
   (c) a small reward/step HUD.

2. `VizBridge` (FastAPI + WebSocket) -- an OPTIONAL live-visualization layer
   for the Three.js frontend under `assets/`. It is never imported by
   training code; it's launched standalone via `scripts/serve_viz.py`, and
   the env only needs to call `VizBridge.publish(env)` if a bridge instance
   exists. Training and `main.py` work identically with or without it.

JSON schema published over the WebSocket (one message per env.step()):
{
  "probe": {"position": [x,y,z], "forward": [x,y,z], "right": [x,y,z], "up": [x,y,z]},
  "target": "head" | "abdomen" | "femur" | null,
  "target_point": [x,y,z] | null,  # the target plane's real 3D point, for the viz's target marker
  "acquired": ["head", ...],
  "reward": float,
  "alpha_deg": float,
  "d_m": float,
  "flag": "AGA" | "SGA" | null,
  "slice_png_base64": "<base64-encoded PNG of the current slice image>",
  "step": int,
  "policy_label": str | null,  # e.g. "random actions" / "trained model (ppo)" -- set via VizBridge.policy_label
  "success": bool,  # True only on the exact step an acquisition happened (see custom_env.py's info dict)
  "episode_ended": bool,  # True on the exact step the episode terminated OR truncated -- lets the viz show an episode-end indicator
  "confirmation": {"png_base64": str, "label": str} | null  # OPTIONAL, only attached by scripts/serve_viz.py
    # right after a success: a SEPARATE reference image (not the literal probe
    # feed) showing what the acquired structure actually looks like, since
    # the real acquisition criterion (angle+depth to the target plane) does
    # not require the structure to be laterally centered in the live slice --
    # see custom_env.py's `_pose_error` docstring / status.md. VizBridge
    # itself never computes this (would need a rendering.py -> custom_env.py
    # dependency it doesn't have); the caller passes it into `publish()`.
}
"""
from __future__ import annotations

import base64
import io
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, WebSocket

from environment.phantom import MATERNAL_ABDOMEN_SEMI_AXES


def render_matplotlib(env, mode: str = "rgb_array") -> Optional[np.ndarray]:
    fig = plt.figure(figsize=(12, 4.5))

    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    _draw_abdomen(ax3d)
    _draw_probe(ax3d, env)
    ax3d.set_title("Probe on abdomen")

    ax_img = fig.add_subplot(1, 3, 2)
    img = env._last_image if env._last_image is not None else np.zeros((128, 128))
    ax_img.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax_img.set_title(f"Slice (target={env.targets[env.target_idx] if env.target_idx < len(env.targets) else 'done'})")
    ax_img.axis("off")

    ax_hud = fig.add_subplot(1, 3, 3)
    ax_hud.axis("off")
    info = env._last_reward_info
    lines = [
        f"step: {env.total_steps}",
        f"target: {info.get('target')}",
        f"alpha: {info.get('alpha_deg', 0):.1f} deg",
        f"d: {info.get('d_m', 0) * 1000:.1f} mm",
        f"acquired: {info.get('acquired', [])}",
        f"flag: {info.get('flag')}",
    ]
    ax_hud.text(0.05, 0.95, "\n".join(lines), va="top", family="monospace")

    fig.tight_layout()

    if mode == "human":
        fig.canvas.draw()
        plt.show(block=False)
        plt.pause(0.001)
        plt.close(fig)
        return None

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    arr = buf[..., :3].copy()
    plt.close(fig)
    return arr


def _draw_abdomen(ax):
    a, b, c = MATERNAL_ABDOMEN_SEMI_AXES
    u = np.linspace(0, np.pi / 2, 20)
    v = np.linspace(0, 2 * np.pi, 20)
    uu, vv = np.meshgrid(u, v)
    x = a * np.sin(uu) * np.cos(vv)
    y = b * np.sin(uu) * np.sin(vv)
    z = c * np.cos(uu)
    ax.plot_surface(x, y, z, alpha=0.15, color="peachpuff")
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.set_zlim(0, 0.15)


def _draw_probe(ax, env):
    pos, forward, right, up = env._probe_frame()[:4]
    ax.scatter(*pos, color="black", s=40)
    tip = pos + forward * 0.05
    ax.plot([pos[0], tip[0]], [pos[1], tip[1]], [pos[2], tip[2]], color="red")

    target_name = env.targets[env.target_idx] if env.target_idx < len(env.targets) else None
    if target_name is not None and env.phantom is not None:
        pt = env.phantom.plane_targets[target_name]
        ax.scatter(*pt.point, color="blue", s=30, marker="x")


def encode_image_base64(image: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class VizBridge:
    """Optional FastAPI + WebSocket bridge streaming env state to the
    Three.js frontend in `assets/`. Never imported by training code --
    only used by `scripts/serve_viz.py`.

    NOTE: `FastAPI`/`WebSocket` must be imported at MODULE level (see top of
    this file), not locally inside `__init__`. This module has `from
    __future__ import annotations`, which stringifies the `websocket:
    WebSocket` type hint below; FastAPI resolves that string via the
    endpoint function's `__globals__` (this module's globals), not via
    whatever was in scope inside `__init__`. A local import here previously
    caused FastAPI to silently fail to recognize the WebSocket parameter --
    it instead required "websocket" as a normal query parameter, so every
    connection was rejected with a validation error. Caught by
    `tests/test_viz_bridge.py`."""

    def __init__(self):
        self.app = FastAPI()
        self._clients: list = []
        # Optional, purely for display: which policy is driving the demo
        # (e.g. "random actions" or "trained model (ppo)"). Set by the
        # caller (scripts/serve_viz.py) after construction -- VizBridge
        # itself has no idea what's stepping the env. None = omit/blank.
        self.policy_label: str | None = None
        # Optional callback(text: str), invoked for every text message any
        # connected client sends over the WebSocket -- previously received
        # messages were read and silently discarded (only to detect
        # disconnection). Settable by a caller (e.g. scripts/serve_viz.py)
        # to turn this into a two-way channel, e.g. a browser keypress
        # sending "resume" to un-pause an episode-end wait. None = still
        # just discard (original behavior, unchanged if unset).
        self.on_client_message = None
        # Last payload sent to any client, if any -- replayed to a NEWLY
        # connecting client immediately on accept (below). Without this, a
        # browser tab that connects while the demo loop is paused waiting
        # for "resume" (see scripts/serve_viz.py's PAUSE_ON_EPISODE_END)
        # sees nothing at all: no new step is coming until someone sends
        # resume, but a fresh page load has no idea the server is paused or
        # what the last state was (e.g. a reload after the tab that
        # originally triggered the pause is gone) -- it just sits at
        # "connected" forever with an empty scene. Replaying the last frame
        # means the new tab immediately sees the real last state, including
        # `episode_ended` (so the pause hint shows) and can un-pause it
        # itself by pressing SPACE/R.
        self._last_payload: dict | None = None

        @self.app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket):
            await websocket.accept()
            self._clients.append(websocket)
            if self._last_payload is not None:
                await websocket.send_json(self._last_payload)
            try:
                while True:
                    text = await websocket.receive_text()
                    if self.on_client_message is not None:
                        self.on_client_message(text)
            except Exception:
                pass
            finally:
                if websocket in self._clients:
                    self._clients.remove(websocket)

    async def publish(self, env, confirmation: dict | None = None) -> dict:
        pos, forward, right, up, _ = env._probe_frame()
        info = env._last_reward_info
        target_name = info.get("target")
        target_point = None
        if target_name is not None and env.phantom is not None:
            target_point = env.phantom.plane_targets[target_name].point.tolist()
        payload = {
            "probe": {
                "position": pos.tolist(), "forward": forward.tolist(),
                "right": right.tolist(), "up": up.tolist(),
            },
            "target": target_name,
            "target_point": target_point,
            "acquired": info.get("acquired", []),
            "reward": info.get("reward", 0.0),
            "alpha_deg": info.get("alpha_deg", 0.0),
            "d_m": info.get("d_m", 0.0),
            "flag": info.get("flag"),
            "slice_png_base64": encode_image_base64(env._last_image) if env._last_image is not None else "",
            "step": env.total_steps,
            "policy_label": self.policy_label,
            "success": info.get("success", False),
            "episode_ended": bool(getattr(env, "_last_terminated", False) or getattr(env, "_last_truncated", False)),
            "confirmation": confirmation,
        }
        self._last_payload = payload
        dead = []
        for client in self._clients:
            try:
                await client.send_json(payload)
            except Exception:
                dead.append(client)
        for d in dead:
            self._clients.remove(d)
        return payload
