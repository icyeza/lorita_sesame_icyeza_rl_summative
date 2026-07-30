"""Phase 1 (non-training) + Phase 2 (conditional widening-schedule train):
does the Arm 2 curriculum-trained policy generalize from its narrow
training start-distribution out to the full, uniform-random start
distribution the real task uses?

Phase 1 loads the ALREADY-TRAINED Arm 2 model (models/
three_arm_exploration_fix/arm2_start_curriculum/model.zip) with NO further
training, and evaluates it deterministically across a sweep of
start-distances, from the exact curriculum radius it trained on
(start_curriculum_max_random_steps=8) out to the original uniform-random
start (start_curriculum=False). Reports success rate / terminal alpha /
terminal d as a function of start-distance.

Phase 2 (ONLY runs if Phase 1 shows a generalization gap) trains a FRESH
PPO policy with a start-radius WIDENING SCHEDULE: begins at the narrow
curriculum radius and widens in stages as training progresses, targeting
the full uniform-random distribution by the end. Uses a deterministic,
timestep-triggered schedule (not an adaptive "widen only once success
holds" controller -- documented simplification, see
StartCurriculumWideningCallback's docstring) via a runtime setter
(`UltrasoundProbeEnv.set_start_curriculum`) called through
`VecEnv.env_method` so all `SubprocVecEnv` workers widen together.

Env config throughout: femur-only, alpha_tol_deg=18, tilt_step_deg=3.0,
shaping_mode="multiplicative" (NOT hybrid -- rejected last pass, it
regressed head/abdomen). d_tol, geometry, features, actuator clamp,
classifier untouched.

Usage: uv run python scripts/generalization_check.py [--phase2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from environment.custom_env import UltrasoundProbeEnv
from training.dqn_training import make_vec_env

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "three_arm_exploration_fix" / "arm2_start_curriculum" / "model.zip"
OUT_DIR = REPO_ROOT / "logs" / "generalization_check"

COMMON_ENV_KWARGS = dict(single_target=True, single_target_which="femur",
                          alpha_tol_deg=18.0, tilt_step_deg=3.0, shaping_mode="multiplicative")

N_EVAL_EPISODES = 100
SEED_START = 9000
HOLDS_UP_SUCCESS_FLOOR = 0.85

LEVELS = {
    "curriculum-narrow (matches training, max_steps=8)": dict(start_curriculum=True, start_curriculum_max_random_steps=8),
    "small (max_steps=20)": dict(start_curriculum=True, start_curriculum_max_random_steps=20),
    "medium (max_steps=40)": dict(start_curriculum=True, start_curriculum_max_random_steps=40),
    "large (max_steps=80)": dict(start_curriculum=True, start_curriculum_max_random_steps=80),
    "uniform-random (full task, start_curriculum=False)": dict(start_curriculum=False),
}

# --- Phase 2: widening-schedule train ---
PHASE2_TOTAL_TIMESTEPS = 120_000
PHASE2_LOG_DIR = REPO_ROOT / "logs" / "generalization_check" / "phase2_widening_train"
PHASE2_MODEL_DIR = REPO_ROOT / "models" / "generalization_check" / "phase2_widening_train"
# Deterministic, timestep-triggered widening schedule -- 4 stages of 30k
# steps each, ending at the full uniform-random distribution. NOT an
# adaptive "widen only once success holds" controller (that needs live
# success-rate monitoring across workers and a more complex controller);
# documented simplification for this pass.
PHASE2_SCHEDULE = [
    (0, dict(start_curriculum=True, max_random_steps=8)),
    (30_000, dict(start_curriculum=True, max_random_steps=40)),
    (60_000, dict(start_curriculum=True, max_random_steps=80)),
    (90_000, dict(start_curriculum=False, max_random_steps=None)),
]


class StartCurriculumWideningCallback(BaseCallback):
    """Advances `PHASE2_SCHEDULE` by timestep, calling
    `UltrasoundProbeEnv.set_start_curriculum` on every SubprocVecEnv worker
    via `env_method` whenever `num_timesteps` crosses a schedule
    breakpoint. Deterministic/timestep-triggered, not adaptive -- see
    module docstring."""

    def __init__(self, schedule: list[tuple[int, dict]], verbose: int = 0):
        super().__init__(verbose)
        self.schedule = schedule
        self._stage_idx = -1

    def _on_step(self) -> bool:
        next_idx = self._stage_idx
        for i, (threshold, _) in enumerate(self.schedule):
            if self.num_timesteps >= threshold:
                next_idx = i
        if next_idx != self._stage_idx:
            self._stage_idx = next_idx
            _, params = self.schedule[self._stage_idx]
            self.training_env.env_method(
                "set_start_curriculum", params["start_curriculum"], params["max_random_steps"],
            )
            if self.verbose:
                print(f"[widening schedule] timestep={self.num_timesteps}: "
                      f"stage {self._stage_idx} -> {params}")
        return True


def evaluate(model, level_env_kwargs: dict, n: int = N_EVAL_EPISODES, seed_start: int = SEED_START):
    env = UltrasoundProbeEnv(seed=1, **COMMON_ENV_KWARGS, **level_env_kwargs)
    successes, terminal_alphas, terminal_ds, freezes, steps_list = [], [], [], [], []
    for i in range(n):
        obs, info = env.reset(seed=seed_start + i)
        done = False
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            steps += 1
        successes.append(bool(info["success"]))
        terminal_alphas.append(info["alpha_deg"])
        terminal_ds.append(info["d_m"] * 1000.0)
        freezes.append(bool(info["freeze_attempted"]))
        steps_list.append(steps)

    successes = np.array(successes)
    terminal_alphas = np.array(terminal_alphas)
    terminal_ds = np.array(terminal_ds)
    freezes = np.array(freezes)
    return dict(
        n=n, success_rate=float(successes.mean()),
        median_terminal_alpha_deg=float(np.median(terminal_alphas)),
        mean_terminal_alpha_deg=float(np.mean(terminal_alphas)),
        median_terminal_d_mm=float(np.median(terminal_ds)),
        mean_terminal_d_mm=float(np.mean(terminal_ds)),
        freeze_attempted_rate=float(freezes.mean()),
        median_steps=float(np.median(steps_list)),
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading trained Arm 2 (start-curriculum) policy from {MODEL_PATH}")
    model = PPO.load(str(MODEL_PATH))

    print(f"\n=== Phase 1: success-vs-start-distance (deterministic eval, N={N_EVAL_EPISODES}/level, NO training) ===")
    results = {}
    for label, kwargs in LEVELS.items():
        r = evaluate(model, kwargs)
        results[label] = r
        print(f"[{label}]")
        print(f"  success_rate={r['success_rate']:.4f}, median_terminal_alpha={r['median_terminal_alpha_deg']:.2f}deg, "
              f"median_terminal_d={r['median_terminal_d_mm']:.2f}mm, freeze_attempted_rate={r['freeze_attempted_rate']:.4f}, "
              f"median_steps={r['median_steps']:.1f}")

    print("\n=== Success-vs-start-distance table ===")
    header = f"{'level':<52} {'success':>9} {'term_alpha_deg':>16} {'term_d_mm':>11} {'freeze_rate':>12}"
    print(header)
    for label, r in results.items():
        print(f"{label:<52} {r['success_rate']:>9.4f} {r['median_terminal_alpha_deg']:>16.2f} "
              f"{r['median_terminal_d_mm']:>11.2f} {r['freeze_attempted_rate']:>12.4f}")

    uniform_key = "uniform-random (full task, start_curriculum=False)"
    uniform_success = results[uniform_key]["success_rate"]
    holds_up = uniform_success >= HOLDS_UP_SUCCESS_FLOOR

    print(f"\n=== READ ===")
    if holds_up:
        print(f"Uniform-random success_rate={uniform_success:.4f} >= {HOLDS_UP_SUCCESS_FLOOR} floor -- "
              f"the policy ALREADY GENERALIZES to the full start distribution.")
        print("RECOMMENDATION: lock the environment at uniform-random starts; Phase 2 is NOT needed.")
    else:
        # find the widest level (in dict order, narrow->wide) where success still holds
        widest_holding = None
        for label, r in results.items():
            if r["success_rate"] >= HOLDS_UP_SUCCESS_FLOOR:
                widest_holding = label
        print(f"Uniform-random success_rate={uniform_success:.4f} < {HOLDS_UP_SUCCESS_FLOOR} floor -- "
              f"GENERALIZATION GAP found.")
        print(f"Widest level where success still holds >= {HOLDS_UP_SUCCESS_FLOOR}: {widest_holding}")
        print("Proceeding to Phase 2 (widening-schedule train) is indicated.")

    with open(OUT_DIR / "phase1_results.json", "w") as f:
        json.dump(dict(levels=results, uniform_success_rate=uniform_success, holds_up=holds_up), f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'phase1_results.json'}")
    return holds_up


def run_phase2():
    PHASE2_LOG_DIR.mkdir(parents=True, exist_ok=True)
    PHASE2_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(**COMMON_ENV_KWARGS, start_curriculum=True, start_curriculum_max_random_steps=8)
    info_keywords = ("success", "freeze_attempted", "d_m", "alpha_deg")

    print(f"\n=== Phase 2: widening-schedule PPO train ===")
    print(f"  schedule (timestep -> params): {PHASE2_SCHEDULE}")
    print(f"  PPO, femur-only, alpha_tol_deg=18, tilt_step_deg=3.0, shaping_mode=multiplicative, "
          f"ent_coef=0.05, n_envs=4, total_timesteps={PHASE2_TOTAL_TIMESTEPS}, seed=0, uncapped")

    env = make_vec_env(env_kwargs, str(PHASE2_LOG_DIR), seed=0, n_envs=4, info_keywords=info_keywords)
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4, gamma=0.99, n_steps=128, batch_size=64, n_epochs=10,
        ent_coef=0.05, gae_lambda=0.95, clip_range=0.2,
        policy_kwargs=dict(net_arch=[64, 64]), seed=0, verbose=0,
    )
    model.set_logger(configure(str(PHASE2_LOG_DIR), ["csv", "tensorboard"]))
    widening_cb = StartCurriculumWideningCallback(PHASE2_SCHEDULE, verbose=1)
    model.learn(total_timesteps=PHASE2_TOTAL_TIMESTEPS, progress_bar=False, callback=widening_cb)

    save_path = PHASE2_MODEL_DIR / "model.zip"
    model.save(str(save_path))
    print(f"Phase 2 training complete. Model saved to {save_path}")

    # training trajectories
    monitor_paths = sorted(PHASE2_LOG_DIR.glob("monitor*.csv"))
    frames = []
    for p in monitor_paths:
        df = pd.read_csv(p, skiprows=1)
        if "r" in df.columns and len(df) > 0:
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True).sort_values("t").reset_index(drop=True)
    combined["cum_timesteps"] = combined["l"].cumsum()
    combined["d_mm"] = combined["d_m"] * 1000.0
    combined.to_csv(PHASE2_LOG_DIR / "episodes.csv", index=False)

    n = len(combined)
    rolling_d = combined["d_mm"].rolling(50, min_periods=1).median()
    rolling_alpha = combined["alpha_deg"].rolling(50, min_periods=1).median()
    rolling_success = combined["success"].astype(float).rolling(50, min_periods=1).mean()
    rolling_freeze = combined["freeze_attempted"].astype(float).rolling(50, min_periods=1).mean()
    fig, axes = plt.subplots(4, 1, figsize=(10, 15), sharex=True)
    axes[0].plot(combined["cum_timesteps"], rolling_d, color="steelblue", linewidth=2)
    axes[0].axhline(12.0, color="red", linestyle="--", label="d_tol=12mm")
    axes[0].set_title("Terminal d vs timesteps"); axes[0].legend()
    axes[1].plot(combined["cum_timesteps"], rolling_alpha, color="purple", linewidth=2)
    axes[1].axhline(18.0, color="red", linestyle="--", label="alpha_tol=18deg")
    axes[1].set_title("Terminal alpha vs timesteps"); axes[1].legend()
    axes[2].plot(combined["cum_timesteps"], rolling_success, color="green", linewidth=2)
    axes[2].set_ylim(-0.02, 1.02); axes[2].set_title("Success rate vs timesteps")
    axes[3].plot(combined["cum_timesteps"], rolling_freeze, color="orange", linewidth=2)
    axes[3].set_ylim(-0.02, 1.02); axes[3].set_title("Freeze-attempt fraction vs timesteps")
    for t, _ in PHASE2_SCHEDULE[1:]:
        for ax in axes:
            ax.axvline(t, color="gray", linestyle=":", alpha=0.6)
    fig.suptitle(f"Phase 2 widening-schedule train (N={n} episodes)\n"
                 f"gray dotted lines = schedule stage transitions", fontsize=10)
    fig.tight_layout()
    fig.savefig(PHASE2_LOG_DIR / "training_trajectories.png", dpi=130)
    plt.close(fig)

    print(f"\n=== Phase 1 RE-RUN on the newly-trained (post-widening) policy ===")
    results = {}
    for label, kwargs in LEVELS.items():
        r = evaluate(model, kwargs)
        results[label] = r
        print(f"[{label}] success_rate={r['success_rate']:.4f}, "
              f"median_terminal_alpha={r['median_terminal_alpha_deg']:.2f}deg, "
              f"median_terminal_d={r['median_terminal_d_mm']:.2f}mm")

    print("\n=== Success-vs-start-distance table (POST-WIDENING) ===")
    header = f"{'level':<52} {'success':>9} {'term_alpha_deg':>16} {'term_d_mm':>11}"
    print(header)
    for label, r in results.items():
        print(f"{label:<52} {r['success_rate']:>9.4f} {r['median_terminal_alpha_deg']:>16.2f} "
              f"{r['median_terminal_d_mm']:>11.2f}")

    with open(PHASE2_LOG_DIR / "phase1_rerun_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2", action="store_true", help="force Phase 2 even if Phase 1 held up")
    args = parser.parse_args()
    holds_up = main()
    if args.phase2 or not holds_up:
        run_phase2()
    else:
        print("\nPhase 1 already holds up -- skipping Phase 2 (pass --phase2 to force).")
