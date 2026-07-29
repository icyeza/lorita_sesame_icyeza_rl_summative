"""Action-granularity <-> tolerance co-calibration sweep + oracle gate.

Addendum 11 precisely localized the residual 30% oracle failure: those
episodes have terminal d near-perfect but terminal alpha parked at
15.2-19.4deg -- just outside the 15deg tolerance -- burning the entire
step budget regardless of how much budget is available. Diagnosis: a 3deg
tilt increment cannot SETTLE inside a 15deg-wide tolerance window from
certain approach angles (e.g. from 17deg, +/-3deg overshoots to 14deg or
20deg, never landing inside [-15,15]). This is an action-granularity vs
tolerance MISMATCH, not a budget or start-distribution problem.

This script sweeps five (alpha_tol_deg, tilt_step_deg) combinations with
the SAME non-training oracle (reused unchanged from
diagnose_scripted_policy.py), picks the tightest-tolerance variant that
clears a ~90%-with-margin gate, and -- ONLY if the gate passes -- runs one
40k-step PPO smoke train.

Permitted changes this pass: alpha_tol_deg and/or tilt_step_deg only.
d_tol, geometry, features, actuator clamp, classifier, shaping_mode
("multiplicative", kept from the prior pass), POTENTIAL_COUPLE_WEIGHT, and
subtask/episode step budgets are all left untouched.

Usage: uv run python scripts/co_calibration_gate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment.custom_env import UltrasoundProbeEnv
from scripts.diagnose_scripted_policy import greedy_episode, TARGET
from training.pg_training import train_ppo

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "logs" / "co_calibration_gate"
SMOKE_LOG_DIR = OUT_DIR / "smoke_train"
SMOKE_MODEL_DIR = REPO_ROOT / "models" / "co_calibration_gate" / "smoke_train"

SHAPING_MODE = "multiplicative"  # kept from the prior two passes
SUBTASK_MAX_STEPS = 150          # kept from Addendum 10/11 -- budget is ruled out as the lever
N_ORACLE_EPISODES = 100
ORACLE_SEED = 42

VARIANTS = {
    "baseline":        dict(alpha_tol_deg=15.0, tilt_step_deg=3.0),
    "widen-tol-18":    dict(alpha_tol_deg=18.0, tilt_step_deg=3.0),
    "widen-tol-20":    dict(alpha_tol_deg=20.0, tilt_step_deg=3.0),
    "finer-step":      dict(alpha_tol_deg=15.0, tilt_step_deg=1.5),
    "finer-step-mid":  dict(alpha_tol_deg=15.0, tilt_step_deg=2.0),
}

GATE_SUCCESS_FLOOR = 0.90
GATE_MARGIN_STEP_FRACTION = 0.8  # median steps-to-success <= 80% of the 150 budget (=120)


def run_oracle(alpha_tol_deg: float, tilt_step_deg: float, n: int = N_ORACLE_EPISODES, seed: int = ORACLE_SEED):
    env = UltrasoundProbeEnv(single_target=True, single_target_which=TARGET, seed=seed,
                              alpha_tol_deg=alpha_tol_deg, tilt_step_deg=tilt_step_deg,
                              subtask_max_steps=SUBTASK_MAX_STEPS, shaping_mode=SHAPING_MODE)
    results = [greedy_episode(env, max_steps=SUBTASK_MAX_STEPS + 20) for _ in range(n)]
    n_success = sum(r["success"] for r in results)
    outcomes = {}
    for r in results:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    n_timeout = outcomes.get("timeout_subtask", 0) + outcomes.get("timeout_episode", 0)
    steps_success = [r["steps"] for r in results if r["success"]]
    terminal_alpha = np.array([r["terminal_alpha_deg"] for r in results])

    stats = dict(
        alpha_tol_deg=alpha_tol_deg, tilt_step_deg=tilt_step_deg, n=n,
        success_rate=n_success / n, timeout_rate=n_timeout / n,
        median_steps_to_success=float(np.median(steps_success)) if steps_success else None,
        median_terminal_alpha_deg=float(np.median(terminal_alpha)),
        outcomes=outcomes,
    )
    print(f"  alpha_tol={alpha_tol_deg}deg, tilt_step={tilt_step_deg}deg: "
          f"success={stats['success_rate']:.4f} ({n_success}/{n}), "
          f"timeout={stats['timeout_rate']:.4f}, "
          f"median steps-to-success={stats['median_steps_to_success']}, "
          f"median terminal alpha={stats['median_terminal_alpha_deg']:.2f}deg, outcomes={outcomes}")
    return stats


def run_smoke_train(alpha_tol_deg: float, tilt_step_deg: float):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SMOKE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    SMOKE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(single_target=True, single_target_which=TARGET,
                       alpha_tol_deg=alpha_tol_deg, tilt_step_deg=tilt_step_deg,
                       subtask_max_steps=SUBTASK_MAX_STEPS, shaping_mode=SHAPING_MODE)
    config = dict(entropy_coef=0.05)
    info_keywords = ("success", "freeze_attempted", "d_m", "alpha_deg")

    print(f"\nSmoke train: PPO, femur-only, alpha_tol_deg={alpha_tol_deg}, "
          f"tilt_step_deg={tilt_step_deg}, subtask_max_steps={SUBTASK_MAX_STEPS}, "
          f"shaping_mode={SHAPING_MODE}, ent_coef=0.05, n_envs=4, total_timesteps=40000, "
          f"seed=0, uncapped")

    model, save_path = train_ppo(
        config, str(SMOKE_LOG_DIR), str(SMOKE_MODEL_DIR), seed=0,
        total_timesteps=40_000, env_kwargs=env_kwargs,
        n_envs=4, max_wall_clock_seconds=None, info_keywords=info_keywords,
    )
    print(f"Smoke train complete. Model saved to {save_path}")

    monitor_paths = sorted(SMOKE_LOG_DIR.glob("monitor*.csv"))
    frames = []
    for p in monitor_paths:
        df = pd.read_csv(p, skiprows=1)
        if "r" in df.columns and len(df) > 0:
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True).sort_values("t").reset_index(drop=True)
    combined["cum_timesteps"] = combined["l"].cumsum()
    combined["d_mm"] = combined["d_m"] * 1000.0
    combined.to_csv(OUT_DIR / "smoke_episodes.csv", index=False)

    n = len(combined)
    last_quarter = combined.iloc[3 * n // 4:]
    first_quarter = combined.iloc[:n // 4]
    overall_success = float(combined["success"].mean())
    overall_freeze = float(combined["freeze_attempted"].mean())
    late_success = float(last_quarter["success"].mean())
    late_freeze = float(last_quarter["freeze_attempted"].mean())
    late_d = float(last_quarter["d_mm"].median())
    first_d = float(first_quarter["d_mm"].median())
    late_alpha = float(last_quarter["alpha_deg"].median())
    first_alpha = float(first_quarter["alpha_deg"].median())

    print(f"\nSmoke train (N={n} episodes):")
    print(f"  Overall: success={overall_success:.4f}, freeze={overall_freeze:.4f}")
    print(f"  First quarter: median terminal d={first_d:.2f}mm, median terminal alpha={first_alpha:.2f}deg")
    print(f"  Last quarter: success={late_success:.4f}, freeze={late_freeze:.4f}, "
          f"median terminal d={late_d:.2f}mm, median terminal alpha={late_alpha:.2f}deg")

    learning_signal = (late_success > 0.0) or (late_d < first_d * 0.9) or (late_alpha < first_alpha * 0.9)
    verdict = "LEARNING SIGNAL PRESENT" if learning_signal else "FLAT AT 40k"
    print(f"  VERDICT: {verdict}")

    rolling_d = combined["d_mm"].rolling(50, min_periods=1).median()
    rolling_alpha = combined["alpha_deg"].rolling(50, min_periods=1).median()
    rolling_success = combined["success"].astype(float).rolling(50, min_periods=1).mean()
    rolling_freeze = combined["freeze_attempted"].astype(float).rolling(50, min_periods=1).mean()

    fig, axes = plt.subplots(4, 1, figsize=(10, 15), sharex=True)
    axes[0].plot(combined["cum_timesteps"], rolling_d, color="steelblue", linewidth=2)
    axes[0].axhline(12.0, color="red", linestyle="--", label="d_tol=12mm")
    axes[0].set_ylabel("rolling median terminal d (mm), window=50")
    axes[0].set_title("Terminal d vs timesteps")
    axes[0].legend()
    axes[1].plot(combined["cum_timesteps"], rolling_alpha, color="purple", linewidth=2)
    axes[1].axhline(alpha_tol_deg, color="red", linestyle="--", label=f"alpha_tol={alpha_tol_deg}deg")
    axes[1].set_ylabel("rolling median terminal alpha (deg), window=50")
    axes[1].set_title("Terminal alpha vs timesteps")
    axes[1].legend()
    axes[2].plot(combined["cum_timesteps"], rolling_success, color="green", linewidth=2)
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_ylabel("rolling success rate, window=50")
    axes[2].set_title("Success rate vs timesteps")
    axes[3].plot(combined["cum_timesteps"], rolling_freeze, color="orange", linewidth=2)
    axes[3].set_ylim(-0.02, 1.02)
    axes[3].set_xlabel("timesteps")
    axes[3].set_ylabel("rolling freeze-attempt fraction, window=50")
    axes[3].set_title("Freeze-attempt fraction vs timesteps")
    fig.suptitle(f"Co-calibration smoke train -- alpha_tol={alpha_tol_deg}deg, "
                 f"tilt_step={tilt_step_deg}deg, shaping_mode={SHAPING_MODE}, 40000 steps\n"
                 f"VERDICT: {verdict}", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "smoke_trajectories.png", dpi=130)
    plt.close(fig)

    summary = dict(
        alpha_tol_deg=alpha_tol_deg, tilt_step_deg=tilt_step_deg,
        subtask_max_steps=SUBTASK_MAX_STEPS, shaping_mode=SHAPING_MODE, ent_coef=0.05,
        n_envs=4, total_timesteps=40_000, seed=0, n_episodes=n,
        overall_success_rate=overall_success, overall_freeze_attempted_rate=overall_freeze,
        late_success_rate=late_success, late_freeze_attempted_rate=late_freeze,
        late_median_terminal_d_mm=late_d, first_quarter_median_terminal_d_mm=first_d,
        late_median_terminal_alpha_deg=late_alpha, first_quarter_median_terminal_alpha_deg=first_alpha,
        verdict=verdict,
    )
    with open(OUT_DIR / "smoke_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Co-calibration oracle sweep (non-training, N=100/variant) ===")
    sweep = {}
    for name, params in VARIANTS.items():
        print(f"[{name}]")
        sweep[name] = run_oracle(**params)

    print("\n=== Co-calibration table ===")
    header = f"{'variant':<16} {'alpha_tol':>10} {'tilt_step':>10} {'success':>9} {'timeout':>9} {'median_term_alpha':>18}"
    print(header)
    for name, s in sweep.items():
        print(f"{name:<16} {s['alpha_tol_deg']:>10.1f} {s['tilt_step_deg']:>10.1f} "
              f"{s['success_rate']:>9.4f} {s['timeout_rate']:>9.4f} {s['median_terminal_alpha_deg']:>18.2f}")

    # Pick the tightest-tolerance variant that clears the gate. "Tightest"
    # ranks baseline(15,3) < finer-step(15,1.5)/finer-step-mid(15,2) (same
    # tolerance, finer action) < widen-tol-18 < widen-tol-20 (looser
    # tolerance). Prefer a finer step at 15deg over ANY tolerance widening
    # if it clears the gate, per this pass's "tightest achievable, not
    # loosest" instruction.
    preference_order = ["baseline", "finer-step", "finer-step-mid", "widen-tol-18", "widen-tol-20"]
    chosen = None
    for name in preference_order:
        s = sweep[name]
        margin_ok = (s["median_steps_to_success"] is not None) and \
                    (s["median_steps_to_success"] <= GATE_MARGIN_STEP_FRACTION * SUBTASK_MAX_STEPS)
        if s["success_rate"] >= GATE_SUCCESS_FLOOR and margin_ok:
            chosen = name
            break

    gate_result = dict(sweep=sweep, chosen=chosen, gate_success_floor=GATE_SUCCESS_FLOOR,
                        gate_margin_step_fraction=GATE_MARGIN_STEP_FRACTION,
                        subtask_max_steps=SUBTASK_MAX_STEPS)
    with open(OUT_DIR / "gate_result.json", "w") as f:
        json.dump(gate_result, f, indent=2)

    if chosen is None:
        print(f"\n=== GATE: FAIL -- no variant reached success>={GATE_SUCCESS_FLOOR} with "
              f"median steps-to-success <= {GATE_MARGIN_STEP_FRACTION*SUBTASK_MAX_STEPS:.0f} ===")
        print("Stopping per hard rule -- no training will run, no escalation beyond alpha_tol/tilt_step.")
        return

    chosen_params = VARIANTS[chosen]
    print(f"\n=== GATE: PASS -- chosen variant = {chosen} "
          f"(alpha_tol={chosen_params['alpha_tol_deg']}deg, tilt_step={chosen_params['tilt_step_deg']}deg) ===")
    print(f"  success={sweep[chosen]['success_rate']:.4f}, "
          f"median steps-to-success={sweep[chosen]['median_steps_to_success']}")
    print(f"  Rationale: tightest tolerance / finest-preferred action set (in preference order "
          f"{preference_order}) that clears success>={GATE_SUCCESS_FLOOR} with margin.")

    print("\nGate passed -- proceeding to the single 40k-step PPO smoke train.")
    run_smoke_train(alpha_tol_deg=chosen_params["alpha_tol_deg"], tilt_step_deg=chosen_params["tilt_step_deg"])


if __name__ == "__main__":
    main()
