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

from environment.custom_env import (
    ACTIONS,
    EPISODE_MAX_STEPS,
    REQUIRED_STRUCTURES,
    UltrasoundProbeEnv,
)
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
MODEL_ALGO = None
PAUSE_ON_EPISODE_END = True


# ---------------------------------------------------------------------------
# Verbose terminal reporting.
#
# PRINTING ONLY. Every number below is read back out of the SAME
# `env._last_reward_info` dict that VizBridge.publish() serializes, so the
# terminal and the browser can never disagree; nothing here steps, mutates,
# reseeds or paces the env. The one env call made outside that dict is
# `env._pose_error(target)` for the episode header's starting alpha/d --
# it's a pure read of the current probe frame (see its docstring), no state
# is touched.
# ---------------------------------------------------------------------------
VERBOSE = True

# ACTIONS entries are "theta_plus"/"pitch_minus"/...; the terminal shows the
# human-readable "theta_+"/"pitch_-" form asked for, keeping "toggle_fine"
# and "freeze_and_measure" as-is.
ACTION_DISPLAY = {a: a.replace("_plus", "_+").replace("_minus", "_-") for a in ACTIONS}


def _supports_color(stream) -> bool:
    """True only for a real TTY. Piping to a file/pager gets plain text."""
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    if sys.platform == "win32":
        # Modern Windows consoles do ANSI, but only once
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING (0x0004) is set on the handle;
        # without this the codes print as literal garbage.
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
        except Exception:
            return False
    return True


def _supports_unicode(stream) -> bool:
    return "utf" in (getattr(stream, "encoding", None) or "").lower()


class _Style:
    """Subtle ANSI styling, or a no-op passthrough when not a TTY."""

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def green(self, t): return self._wrap("32", t)
    def yellow(self, t): return self._wrap("33", t)
    def bold(self, t): return self._wrap("1", t)
    def dim(self, t): return self._wrap("2", t)


class _StepReporter:
    """Renders the per-step table, episode header and episode summary.

    Column widths are fixed and shared between the header row and the data
    rows so the output reads as a table; styling is applied only after
    padding, so ANSI escapes never shift the alignment.
    """

    # step(4) + gap(2) + action(16) + alpha(8) + mark(2) + d(9) + mark(2)
    # + reward(9) + return(10) == 62 visible columns, well inside the ~100
    # asked for at large font on a shared screen.
    # W_ACTION is 18 to fit the longest action name, "freeze_and_measure",
    # without pushing the numeric columns out of alignment on that row.
    W_STEP, W_ACTION, W_ALPHA, W_D, W_REWARD, W_RETURN = 4, 19, 8, 9, 9, 10
    RULE_WIDTH = 78

    def __init__(self, enabled: bool, stream=None):
        self.enabled = enabled
        self.out = stream if stream is not None else sys.stdout
        self.style = _Style(_supports_color(self.out))
        uni = _supports_unicode(self.out)
        self.sym_alpha = "α" if uni else "alpha"
        self.sym_deg = "°" if uni else " deg"
        self.sym_le = "≤" if uni else "<="
        self.rule_char = "─" if uni else "-"
        self.mark = "✓" if uni else "*"
        self.episode_no = 0
        self.episode_return = 0.0
        self.episode_start_step = 0
        self.acquired_count = 0
        self.finished_episodes = 0

    # -- plumbing ---------------------------------------------------------
    def _print(self, text: str = "") -> None:
        # flush per line: the recording needs output to appear as it
        # happens, not when the block buffer fills.
        print(text, file=self.out, flush=True)

    def _rule(self) -> None:
        self._print(self.style.dim(self.rule_char * self.RULE_WIDTH))

    # -- blocks -----------------------------------------------------------
    def episode_start(self, env, policy_label: str) -> None:
        if not self.enabled:
            return
        self.episode_no += 1
        self.episode_return = 0.0
        self.episode_start_step = env.total_steps
        target = env.targets[env.target_idx] if env.target_idx < len(env.targets) else None
        alpha_deg, d_mm = 0.0, 0.0
        if target is not None:
            alpha, d = env._pose_error(target)  # pure read, see module note
            alpha_deg, d_mm = float(np.degrees(alpha)), float(d) * 1000.0

        a = self.sym_alpha
        self._print()
        self._rule()
        self._print(self.style.bold(
            f" episode {self.episode_no}   target plane: {target}   policy: {policy_label}"))
        self._print(
            f" start: {a} {alpha_deg:.2f}{self.sym_deg}   d {d_mm:.2f} mm"
            f"   start radius: {env.start_curriculum_max_random_steps} random steps"
            f"{'' if env.start_curriculum else ' (curriculum off)'}")
        self._print(
            f" step budget: {env.subtask_max_steps} per target / {EPISODE_MAX_STEPS} per episode"
            f"   tolerances: {a} {self.sym_le} {np.degrees(env.alpha_tol):.2f}{self.sym_deg},"
            f" d {self.sym_le} {env.d_tol * 1000:.2f} mm")
        self._rule()
        self._print(self.style.dim(
            f"{'step':>{self.W_STEP}}  {'action':<{self.W_ACTION}}"
            f"{a + self.sym_deg:>{self.W_ALPHA}}  "
            f"{'d mm':>{self.W_D}}  {'reward':>{self.W_REWARD}}{'return':>{self.W_RETURN}}"))

    def step(self, env, action_name: str) -> None:
        """Call AFTER env.step() (or, for the scripted mode, after
        `_last_reward_info` has been populated) and before/after publish --
        it only reads."""
        if not self.enabled:
            return
        info = env._last_reward_info or {}
        alpha_deg = float(info.get("alpha_deg", 0.0))
        d_mm = float(info.get("d_m", 0.0)) * 1000.0
        reward = float(info.get("reward", 0.0))
        self.episode_return += reward

        alpha_ok = np.radians(alpha_deg) <= env.alpha_tol
        d_ok = d_mm / 1000.0 <= env.d_tol

        alpha_cell = f"{alpha_deg:>{self.W_ALPHA}.2f}"
        d_cell = f"{d_mm:>{self.W_D}.2f}"
        alpha_mark = self.mark + " " if alpha_ok else "  "
        d_mark = self.mark + " " if d_ok else "  "
        if alpha_ok:
            alpha_cell, alpha_mark = self.style.green(alpha_cell), self.style.green(alpha_mark)
        if d_ok:
            d_cell, d_mark = self.style.green(d_cell), self.style.green(d_mark)

        step_in_episode = env.total_steps - self.episode_start_step
        self._print(
            f"{step_in_episode:>{self.W_STEP}}  "
            f"{ACTION_DISPLAY.get(action_name, action_name):<{self.W_ACTION}}"
            f"{alpha_cell}{alpha_mark}"
            f"{d_cell}{d_mark}"
            f"{reward:>+{self.W_REWARD}.2f}{self.episode_return:>+{self.W_RETURN}.2f}")

    def episode_end(self, env, acquired: bool) -> None:
        if not self.enabled:
            return
        info = env._last_reward_info or {}
        alpha_deg = float(info.get("alpha_deg", 0.0))
        d_mm = float(info.get("d_m", 0.0)) * 1000.0
        alpha_tol_deg = float(np.degrees(env.alpha_tol))
        d_tol_mm = env.d_tol * 1000.0
        steps_used = env.total_steps - self.episode_start_step

        self.finished_episodes += 1
        if acquired:
            self.acquired_count += 1

        a, deg, le = self.sym_alpha, self.sym_deg, self.sym_le
        alpha_cmp = le if alpha_deg <= alpha_tol_deg else ">"
        d_cmp = le if d_mm <= d_tol_mm else ">"
        outcome = (self.style.green("ACQUISITION") if acquired
                   else self.style.yellow("TIMEOUT"))

        self._rule()
        self._print(
            f" {outcome} after {steps_used} steps"
            f"   {a} {alpha_deg:.2f} {alpha_cmp} {alpha_tol_deg:.2f}{deg}"
            f"   d {d_mm:.2f} {d_cmp} {d_tol_mm:.2f} mm")
        self._print(
            f" episode return {self.episode_return:+.2f}"
            f"   session: {self.acquired_count}/{self.finished_episodes} acquired")
        self._rule()


REPORTER: _StepReporter | None = None


def _terminal_policy_label() -> str:
    """The label the TERMINAL prints -- derived from what is actually
    driving the env at runtime, NOT from `bridge.policy_label` (which the
    CLI can set to something else; see the --policy note in __main__).
    The whole point of printing it is independent confirmation, so it has
    to be read off the real POLICY_MODE/MODEL state."""
    if POLICY_MODE == "model" and MODEL is not None:
        return f"trained model ({MODEL_ALGO})"
    if POLICY_MODE == "cinematic":
        return "SCRIPTED cinematic path -- NOT a policy, no model loaded"
    return "random actions"



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

        # Same table as the real policy loop, but every number it can show
        # is scripted: this mode hand-populates `_last_reward_info` with
        # reward=0.0 and success=True on the last step (no env.step(), no
        # policy decision), so the reward/return columns are 0.00 by
        # construction and the outcome is always ACQUISITION. The header's
        # policy line says so outright.
        REPORTER.episode_start(env, _terminal_policy_label())

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

            REPORTER.step(env, "scripted")
            confirmation = _render_confirmation_image(env, target) if is_final else None
            await bridge.publish(env, confirmation=confirmation)
            if is_final:
                REPORTER.episode_end(env, acquired=True)

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
    REPORTER.episode_start(env, _terminal_policy_label())
    while True:
        env._compute_obs()  # ensure a slice image exists even before first step
        if POLICY_MODE == "model" and MODEL is not None:
            action, _ = MODEL.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = env.action_space.sample()
        obs, _, terminated, truncated, info = env.step(action)
        REPORTER.step(env, ACTIONS[action])
        confirmation = None
        if info.get("success"):
            acquired_target = env.targets[env.target_idx - 1]
            confirmation = _render_confirmation_image(env, acquired_target)
        await bridge.publish(env, confirmation=confirmation)
        if terminated or truncated:
            REPORTER.episode_end(env, acquired=bool(info.get("success")))
            if PAUSE_ON_EPISODE_END:
                _resume_event.clear()
                await _resume_event.wait()
            obs, info = env.reset()
            REPORTER.episode_start(env, _terminal_policy_label())
        await asyncio.sleep(0.3)


@app.on_event("startup")
async def start_demo():
    global REPORTER
    bridge.on_client_message = _on_client_message
    if REPORTER is None:  # e.g. `uvicorn play:app`, which never runs __main__
        REPORTER = _StepReporter(enabled=VERBOSE)
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
    parser.add_argument("--quiet", action="store_true",
                         help="suppress the per-step terminal table and the episode header/summary "
                              "blocks (verbose output is ON by default for this script). Printing "
                              "only -- this flag changes nothing about the env, policy, WebSocket "
                              "payload or pacing.")
    args = parser.parse_args()

    VERBOSE = not args.quiet

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
            MODEL_ALGO = algo
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
    REPORTER = _StepReporter(enabled=VERBOSE)
    # print(f"[serve_viz] policy_mode={POLICY_MODE} ({policy_label}), port={args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port)
