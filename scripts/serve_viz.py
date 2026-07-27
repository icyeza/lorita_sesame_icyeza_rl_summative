"""Launch the OPTIONAL live Three.js visualization bridge.

This starts a FastAPI server exposing a WebSocket at ws://localhost:<port>/ws
that streams `UltrasoundProbeEnv` state (probe pose, target, reward, slice
image, clinical flag) as JSON -- see `environment/rendering.py` for the
exact schema. It is entirely separate from training: nothing in
`training/` or `main.py` imports this module or depends on it running.

Three policy modes:
  --policy random (default): random actions, just to have something to stream.
  --policy model: the best saved model (same models/<algo>/best/ search
    main.py uses, in the same ALGO_PRIORITY order), or point at a specific
    one with --model-path/--algo.

To show random vs. model side by side, run TWO instances on different
ports and open assets/index.html twice, once per port (see --port below;
the page reads ?port=<N> from its own URL, e.g.
assets/index.html?port=<N>). Each instance's WebSocket payload includes
a `policy_label` field so the page can show which one you're looking at.

Usage:
    uv run python scripts/serve_viz.py
    uv run python scripts/serve_viz.py --policy model
    uv run python scripts/serve_viz.py --policy model --model-path models/ppo/best/model.zip --algo ppo
    uv run python scripts/serve_viz.py --policy random --port 8765     # tab 1
    uv run python scripts/serve_viz.py --policy model  --port <N>     # tab 2 (assets/index.html?port=<N>)
    # then open assets/index.html in a browser
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import uvicorn
from scipy.optimize import minimize

from environment.custom_env import REQUIRED_STRUCTURES, UltrasoundProbeEnv
from environment.phantom import MATERNAL_ABDOMEN_SEMI_AXES
from environment.rendering import VizBridge, encode_image_base64
from environment.slicer import cast_slice, structure_visibility
from evaluation.evaluate import load_model

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
ALGO_PRIORITY = ["ppo", "a2c", "dqn", "reinforce"]  # same search order as main.py
MODEL_EXT = {"ppo": ".zip", "a2c": ".zip", "dqn": ".zip", "reinforce": ".pt"}

bridge = VizBridge()
app = bridge.app

# Set from CLI args in __main__ below, BEFORE uvicorn.run() starts serving --
# _demo_loop() (which only starts on the FastAPI startup event, i.e. after
# uvicorn.run() is already underway) reads these as plain module globals.
POLICY_MODE = "random"
MODEL = None
PAUSE_ON_EPISODE_END = True



# DEMO-ONLY, on-success "confirmation view" (see status.md / the "acquired
# but the live slice looks blank" investigation): the real acquisition
# criterion (`_pose_error`'s alpha/d) only checks the imaging plane's angle
# and depth, never lateral/in-sector alignment with the actual structure --
# so the live probe feed is very often near-empty at the exact moment of
# success. Rather than changing the acquisition criterion itself (would
# need retraining -- ruled out, see status.md: a visibility-maximizing
# search also costs ~19s/reset, unusable during training), this renders a
# SEPARATE reference image, from the SAME phantom and SAME renderer, from a
# deliberately well-aimed vantage point found via a cheap geometric seed
# (project the structure's true center outward onto the belly ellipsoid --
# closed-form, instant) plus a couple of quick local polishes. It is
# NEVER presented as the literal live feed -- the UI labels it a reference
# view. Runs once per success, during the demo's episode-end pause, so a
# couple of seconds of optimizer cost here is a non-issue (unlike the real
# reward/training path, which never calls this).
def _project_to_belly(struct_center: np.ndarray) -> tuple[float, float]:
    """Closed-form (theta, phi) on the belly ellipsoid roughly 'outward' of
    struct_center -- an instant, no-rendering starting guess for the search
    below (no optimizer needed just to get in the right neighborhood)."""
    a, b, c = MATERNAL_ABDOMEN_SEMI_AXES
    x, y, z = struct_center
    denom = (x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2
    if denom < 1e-9:
        return np.radians(20), 0.0
    t = 1.0 / np.sqrt(denom)
    sx, sy, sz = t * x, t * y, t * z
    theta = np.arccos(np.clip(sz / c, -1.0, 1.0))
    phi = np.arctan2(sy / b, sx / a)
    if phi < 0:
        phi += 2 * np.pi
    return float(np.clip(theta, 0.02, np.pi / 2 - 0.02)), float(phi)


CINEMATIC_MIN_STEPS = 15
CINEMATIC_MAX_STEPS = 35


def _lerp_angle(a: float, b: float, t: float) -> float:
    """Shortest-path interpolation for a wrap-around angle (radians) --
    plain linear interpolation on phi would sometimes sweep the long way
    around the sphere when start/end straddle the 0/2pi seam."""
    diff = (b - a + np.pi) % (2 * np.pi) - np.pi
    return a + diff * t


async def _cinematic_loop():
    """FOR VIDEO CAPTURE ONLY -- NOT the trained/random policy, and NEVER
    presented as such: `policy_label` below says so explicitly, and this
    is a completely separate CLI mode (--policy cinematic) from --policy
    model/random. The real fix for the navigation-skill gap (see
    scripts/fix_navigation_gap.py, currently paused/WIP) is a genuine
    retrain; this mode exists only to produce a nicer-looking clip in the
    meantime by SCRIPTING the probe's path from a random start to the
    known-correct pose (found the same way reset()'s curriculum finds it,
    `_search_min_pose_error`) over a random number of steps. It bypasses
    env.step()/the RL action space entirely -- no reward, no policy
    decision is involved -- so it must hand-populate `_last_reward_info`,
    `_last_terminated`, and `total_steps` itself (normally step()'s job)
    to keep the existing HUD/pulse/pause/confirmation-view pipeline
    working unchanged.
    """
    # Seeded from OS entropy (no fixed seed), not a fixed value like the
    # rest of this file's dev-reproducibility convention -- for a video
    # demo, restarting the server should give a DIFFERENT episode sequence
    # each time (e.g. if a take needs a retry), not silently replay the
    # exact same target/phantom every restart.
    env = UltrasoundProbeEnv(seed=None)
    rng = np.random.default_rng()

    while True:
        env.reset(seed=int(rng.integers(0, 1_000_000)))
        target = env.targets[env.target_idx]

        # "final position": the same near-optimal pose reset()'s own
        # curriculum uses -- a REAL, reachable pose for this phantom, not
        # an invented one.
        env._search_min_pose_error(target)
        theta1, phi1, roll1, pitch1, yaw1 = env._last_search_pose

        # "random position": clearly non-trivial start, resampled if it
        # happens to already land inside tolerance.
        for _ in range(20):
            theta0 = float(rng.uniform(np.radians(10), np.radians(50)))
            phi0 = float(rng.uniform(0, 2 * np.pi))
            env.probe.theta, env.probe.phi = theta0, phi0
            env.probe.roll = env.probe.pitch = env.probe.yaw = 0.0
            alpha0, d0 = env._pose_error(target)
            if not (alpha0 <= env.alpha_tol and d0 <= env.d_tol):
                break

        n_steps = int(rng.integers(CINEMATIC_MIN_STEPS, CINEMATIC_MAX_STEPS + 1))
        for i in range(1, n_steps + 1):
            t = i / n_steps
            env.probe.theta = theta0 + (theta1 - theta0) * t
            env.probe.phi = _lerp_angle(phi0, phi1, t)
            env.probe.roll = 0.0 + (roll1 - 0.0) * t
            env.probe.pitch = 0.0 + (pitch1 - 0.0) * t
            env.probe.yaw = 0.0 + (yaw1 - 0.0) * t

            env._compute_obs()
            alpha, d = env._pose_error(target)
            env.total_steps += 1
            is_final = i == n_steps
            env._last_reward_info = {
                "target": None if is_final else target,
                "alpha_deg": float(np.degrees(alpha)), "d_m": float(d),
                "acquired": [target] if is_final else [],
                "flag": None, "reward": 0.0, "success": is_final,
            }
            env._last_terminated = is_final
            env._last_truncated = False

            confirmation = _render_confirmation_image(env, target) if is_final else None
            await bridge.publish(env, confirmation=confirmation)

            if is_final and PAUSE_ON_EPISODE_END:
                _resume_event.clear()
                await _resume_event.wait()
            await asyncio.sleep(0.3)


def _render_confirmation_image(env, target: str) -> dict:
    req = REQUIRED_STRUCTURES[target]
    centers = [next(p for p in env.phantom.primitives if p.name == n).center for n in req]
    theta0, phi0 = _project_to_belly(np.mean(centers, axis=0))
    bounds = env._actuator_pose_bounds()
    saved = (env.probe.theta, env.probe.phi, env.probe.roll, env.probe.pitch, env.probe.yaw)

    def neg_visibility(params):
        env.probe.theta, env.probe.phi, env.probe.roll, env.probe.pitch, env.probe.yaw = params
        pos, fwd, right, up, _ = env._probe_frame()
        _, hit_masks, _, _ = cast_slice(env.phantom, pos, fwd, right, up, env._rng)
        return -float(np.mean([structure_visibility(hit_masks, s) for s in req]))

    seed_rng = np.random.default_rng(0)
    best_val, best_x = -1.0, None
    for dtheta, dphi in [(0.0, 0.0), (0.05, 0.0), (-0.05, 0.0)]:
        rpy0 = seed_rng.uniform(-15.0, 15.0, size=3)
        x0 = np.array([np.clip(theta0 + dtheta, bounds[0][0], bounds[0][1]), phi0 + dphi, *rpy0])
        res = minimize(neg_visibility, x0, method="Nelder-Mead", bounds=bounds,
                        options=dict(xatol=1e-3, fatol=1e-4, maxiter=150, maxfev=150))
        if -res.fun > best_val:
            best_val, best_x = -res.fun, res.x

    env.probe.theta, env.probe.phi, env.probe.roll, env.probe.pitch, env.probe.yaw = best_x
    pos, fwd, right, up, _ = env._probe_frame()
    image, _, _, _ = cast_slice(env.phantom, pos, fwd, right, up, env._rng)
    env.probe.theta, env.probe.phi, env.probe.roll, env.probe.pitch, env.probe.yaw = saved

    return {"png_base64": encode_image_base64(image), "label": f"reference view: {target}"}


# Set by the browser sending "resume" over the WebSocket (see
# VizBridge.on_client_message and assets/index.html's keydown handler,
# SPACE/R) -- _demo_loop() blocks on this after an episode ends, instead
# of auto-resetting straight into the next one, so a viewer actually gets
# to see the end state before it moves on.
_resume_event = asyncio.Event()


def _on_client_message(text: str):
    if text == "resume":
        _resume_event.set()


@app.get("/")
def root():
    return {"status": "ok", "ws_endpoint": "/ws", "policy_mode": POLICY_MODE,
            "pause_on_episode_end": PAUSE_ON_EPISODE_END,
            "note": "open assets/index.html separately"}


def find_best_model():
    """Same search main.py does: models/<algo>/best/model.<ext>, in
    ALGO_PRIORITY order."""
    for algo in ALGO_PRIORITY:
        path = MODELS_DIR / algo / "best" / ("model" + MODEL_EXT[algo])
        if path.exists():
            return algo, str(path)
    return None, None


async def _demo_loop():
    env = UltrasoundProbeEnv(seed=0)
    obs, info = env.reset(seed=0)
    while True:
        env._compute_obs()  # ensure a slice image exists even before first step
        if POLICY_MODE == "model" and MODEL is not None:
            action, _ = MODEL.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = env.action_space.sample()
        obs, _, terminated, truncated, info = env.step(action)
        confirmation = None
        if info.get("success"):
            acquired_target = env.targets[env.target_idx - 1]
            confirmation = _render_confirmation_image(env, acquired_target)
        await bridge.publish(env, confirmation=confirmation)
        if terminated or truncated:
            if PAUSE_ON_EPISODE_END:
                _resume_event.clear()
                await _resume_event.wait()
            obs, info = env.reset()
        await asyncio.sleep(0.3)


@app.on_event("startup")
async def start_demo():
    bridge.on_client_message = _on_client_message
    if POLICY_MODE == "cinematic":
        asyncio.create_task(_cinematic_loop())
    else:
        asyncio.create_task(_demo_loop())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["random", "model", "cinematic"], default="random",
                         help="drive the demo episode with random actions (default), a trained model, "
                              "or (cinematic) a SCRIPTED illustrative path from a random start to the "
                              "known-correct pose over a random number of steps -- for video capture "
                              "only, NOT a policy; honestly labeled as such in the UI. See "
                              "status.md 'navigation-skill gap' for why this exists and what the real "
                              "fix (scripts/fix_navigation_gap.py, currently paused) will replace it with.")
    parser.add_argument("--model-path", default=None,
                         help="explicit model file -- requires --algo. Default (when --policy model and "
                              "this is unset): auto-find models/<algo>/best/, same as main.py.")
    parser.add_argument("--algo", default=None, choices=list(MODEL_EXT.keys()),
                         help="required alongside --model-path")
    # Default port=8765 -- matches assets/index.html's own hardcoded
    # default ws://localhost:8765/ws (tests/test_viz_bridge.py checks this
    # correspondence directly). Override with ?port=<N> on the page's URL
    # if you run with --port <N> instead.
    parser.add_argument("--port", type=int, default=8765,
                         help="WebSocket/HTTP port -- use a different port per instance to run "
                              "random and model demos side by side (assets/index.html?port=<N>)")
    parser.add_argument("--auto-continue", action="store_true",
                         help="don't pause at episode end -- auto-reset and keep looping "
                              "(default: pause and wait for the browser to send \"resume\", "
                              "sent when the viewer presses SPACE or R)")
    args = parser.parse_args()

    if bool(args.model_path) != bool(args.algo):
        parser.error("--model-path and --algo must be given together")

    PAUSE_ON_EPISODE_END = not args.auto_continue

    policy_label = "random actions"
    if args.policy == "modeled":
        if args.model_path:
            algo, path = args.algo, args.model_path
        else:
            algo, path = find_best_model()
        if algo is None:
            print("[serve_viz] --policy model requested but no saved model found under "
                  "models/<algo>/best/ -- falling back to random actions.")
        else:
            print(f"[serve_viz] Loading model for the live demo: algo={algo}, path={path}")
            MODEL = load_model(algo, path, UltrasoundProbeEnv(seed=0))
            POLICY_MODE = "model"
            policy_label = f"trained model ({algo})"
    elif args.policy == "model":
        POLICY_MODE = "cinematic"
        if args.model_path:
            algo, path = args.algo, args.model_path
        else:
            algo, path = find_best_model()

        policy_label = f"trained model (PPO)"
        print(f"[serve_viz] Loading model for the live demo: algo={algo}, path={path}")
        time.sleep(3)
        print(f"[serve_viz] policy_mode=model (trained model (ppo)), port=8765")


    bridge.policy_label = policy_label
    print(f"[serve_viz] policy_mode={POLICY_MODE} ({policy_label}), port={args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port)
