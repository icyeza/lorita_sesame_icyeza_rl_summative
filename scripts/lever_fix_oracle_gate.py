"""Two-lever fix + oracle-gated smoke train.

The scripted diagnosis (scripts/diagnose_scripted_policy.py) localized the
failure precisely:
  - BUDGET: even a ground-truth oracle greedy controller needed a median 55
    of 60 subtask steps and still failed 65% on timeout -- the budget sat
    at the oracle's own ceiling, leaving no room for a still-improving,
    not-yet-optimal learner to ever succeed.
  - REWARD: at the training settling point (alpha=0.30deg, d=35mm) the
    additive shaping had already banked 74% of max (alpha-term 98%
    saturated, d-term 50%), so perfecting alpha alone paid off almost in
    full regardless of d.

This script applies BOTH levers as EXPERIMENTAL, off-by-default
UltrasoundProbeEnv constructor arguments (`subtask_max_steps`,
`shaping_mode="multiplicative"` via `compute_potential_multiplicative` --
see environment/custom_env.py), then:
  1. Re-runs the NON-TRAINING oracle greedy controller (reused from
     diagnose_scripted_policy.py) as a hard gate.
  2. Re-reads the shaping fraction banked at the settling point under the
     new multiplicative potential.
  3. ONLY IF the gate passes: runs a single short (40k-step) PPO smoke
     train to check for a LEARNING signal (not convergence).

No training happens unless the oracle gate passes. Nothing here changes
UltrasoundProbeEnv's defaults -- subtask_max_steps still defaults to 60,
shaping_mode still defaults to "additive".

Usage: uv run python scripts/lever_fix_oracle_gate.py
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

from environment.custom_env import (
    UltrasoundProbeEnv, POTENTIAL_ALPHA_WEIGHT, POTENTIAL_ALPHA_SCALE,
    POTENTIAL_D_WEIGHT, POTENTIAL_D_SCALE, POTENTIAL_V_WEIGHT, POTENTIAL_COUPLE_WEIGHT,
)
from scripts.diagnose_scripted_policy import greedy_episode, TARGET
from training.pg_training import train_ppo

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "logs" / "lever_fix_oracle_gate"
SMOKE_LOG_DIR = OUT_DIR / "smoke_train"
SMOKE_MODEL_DIR = REPO_ROOT / "models" / "lever_fix_oracle_gate" / "smoke_train"

# --- Lever values (experimental; not committed as new defaults) ---
NEW_SUBTASK_MAX_STEPS = 150
NEW_SHAPING_MODE = "multiplicative"

# --- Oracle gate ---
N_ORACLE_EPISODES = 100
GATE_MIN_SUCCESS_RATE = 0.85  # "aim >= ~85%" per the brief

# --- Smoke train ---
SMOKE_TOTAL_TIMESTEPS = 40_000
SMOKE_N_ENVS = 4
SMOKE_SEED = 0
SMOKE_ENT_COEF = 0.05  # unchanged from the freeze-reward experiment's working entropy setting
ROLLING_WINDOW = 50
D_TOL_MM = 12.0

SETTLE_ALPHA_DEG = 0.30
SETTLE_D_MM = 35.0


def run_oracle_gate(subtask_max_steps: int, shaping_mode: str, n: int = N_ORACLE_EPISODES):
    """Re-runs the exact greedy_episode() oracle from diagnose_scripted_policy.py
    (unchanged logic -- reused, not reimplemented), against an env
    constructed with the two levers applied. shaping_mode does not actually
    affect greedy_episode()'s own action-selection metric (it uses raw
    alpha/alpha_tol + d/d_tol, not the potential), but IS exercised here
    because it changes env.step()'s reward/potential bookkeeping that
    greedy_episode() reads back (info['reward']) -- and, more importantly,
    matching env construction between the oracle check and the eventual
    smoke-train keeps this a true apples-to-apples gate on the SAME env
    configuration PPO would train against."""
    env = UltrasoundProbeEnv(single_target=True, single_target_which=TARGET, seed=42,
                              subtask_max_steps=subtask_max_steps, shaping_mode=shaping_mode)
    results = [greedy_episode(env) for _ in range(n)]
    success_rate = float(np.mean([r["success"] for r in results]))
    terminal_d = np.array([r["terminal_d_mm"] for r in results])
    terminal_alpha = np.array([r["terminal_alpha_deg"] for r in results])
    steps_success = [r["steps"] for r in results if r["success"]]
    outcomes = {}
    for r in results:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1

    stats = dict(
        n=n, subtask_max_steps=subtask_max_steps, shaping_mode=shaping_mode,
        success_rate=success_rate,
        median_steps_to_success=float(np.median(steps_success)) if steps_success else None,
        terminal_d_median_mm=float(np.median(terminal_d)), terminal_d_mean_mm=float(np.mean(terminal_d)),
        terminal_alpha_median_deg=float(np.median(terminal_alpha)),
        terminal_alpha_mean_deg=float(np.mean(terminal_alpha)),
        outcomes=outcomes,
    )
    print(f"  success_rate={success_rate:.4f} ({sum(r['success'] for r in results)}/{n})")
    if steps_success:
        print(f"  median steps-to-success: {np.median(steps_success):.1f} / {subtask_max_steps} budget")
    print(f"  terminal d (mm): median={np.median(terminal_d):.2f} mean={np.mean(terminal_d):.2f}")
    print(f"  terminal alpha (deg): median={np.median(terminal_alpha):.2f} mean={np.mean(terminal_alpha):.2f}")
    print(f"  outcome breakdown: {outcomes}")
    return stats


def reward_readout(shaping_mode: str, alpha_deg: float = SETTLE_ALPHA_DEG, d_mm: float = SETTLE_D_MM):
    alpha = np.radians(alpha_deg)
    d = d_mm / 1000.0
    if shaping_mode == "multiplicative":
        f_alpha = np.exp(-alpha / POTENTIAL_ALPHA_SCALE)
        g_d = np.exp(-d / POTENTIAL_D_SCALE)
        coupled_term = POTENTIAL_COUPLE_WEIGHT * f_alpha * g_d
        total = POTENTIAL_V_WEIGHT * 1.0 + coupled_term  # v=1 (max) to isolate the alpha/d coupling effect
        total_max = POTENTIAL_V_WEIGHT * 1.0 + POTENTIAL_COUPLE_WEIGHT
        return dict(shaping_mode=shaping_mode, f_alpha=float(f_alpha), g_d=float(g_d),
                    coupled_term=float(coupled_term), total=float(total), total_max=float(total_max),
                    pct_banked=100.0 * total / total_max)
    else:
        alpha_term = POTENTIAL_ALPHA_WEIGHT * np.exp(-alpha / POTENTIAL_ALPHA_SCALE)
        d_term = POTENTIAL_D_WEIGHT * np.exp(-d / POTENTIAL_D_SCALE)
        total = POTENTIAL_V_WEIGHT * 1.0 + alpha_term + d_term
        total_max = POTENTIAL_V_WEIGHT * 1.0 + POTENTIAL_ALPHA_WEIGHT + POTENTIAL_D_WEIGHT
        return dict(shaping_mode=shaping_mode, alpha_term=float(alpha_term), d_term=float(d_term),
                    total=float(total), total_max=float(total_max), pct_banked=100.0 * total / total_max)


def run_smoke_train():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SMOKE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    SMOKE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(single_target=True, single_target_which=TARGET,
                       subtask_max_steps=NEW_SUBTASK_MAX_STEPS, shaping_mode=NEW_SHAPING_MODE)
    config = dict(entropy_coef=SMOKE_ENT_COEF)
    info_keywords = ("success", "freeze_attempted", "d_m", "alpha_deg")

    print(f"\nSmoke train: PPO, femur-only, subtask_max_steps={NEW_SUBTASK_MAX_STEPS}, "
          f"shaping_mode={NEW_SHAPING_MODE}, ent_coef={SMOKE_ENT_COEF}, n_envs={SMOKE_N_ENVS}, "
          f"total_timesteps={SMOKE_TOTAL_TIMESTEPS}, seed={SMOKE_SEED}, uncapped")

    model, save_path = train_ppo(
        config, str(SMOKE_LOG_DIR), str(SMOKE_MODEL_DIR), seed=SMOKE_SEED,
        total_timesteps=SMOKE_TOTAL_TIMESTEPS, env_kwargs=env_kwargs,
        n_envs=SMOKE_N_ENVS, max_wall_clock_seconds=None, info_keywords=info_keywords,
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
    overall_success = float(combined["success"].mean())
    overall_freeze = float(combined["freeze_attempted"].mean())
    late_success = float(last_quarter["success"].mean())
    late_freeze = float(last_quarter["freeze_attempted"].mean())
    late_d = float(last_quarter["d_mm"].median())
    first_quarter_d = float(combined.iloc[:n // 4]["d_mm"].median())

    print(f"\nSmoke train (N={n} episodes):")
    print(f"  Overall: success={overall_success:.4f}, freeze={overall_freeze:.4f}")
    print(f"  Last quarter: success={late_success:.4f}, freeze={late_freeze:.4f}, "
          f"median terminal d={late_d:.2f}mm (first quarter: {first_quarter_d:.2f}mm)")

    learning_signal = (late_success > 0.0) or (late_d < first_quarter_d * 0.9)
    verdict = "LEARNING SIGNAL PRESENT" if learning_signal else "FLAT AT 40k"
    print(f"  VERDICT: {verdict}")

    rolling_d = combined["d_mm"].rolling(ROLLING_WINDOW, min_periods=1).median()
    rolling_success = combined["success"].astype(float).rolling(ROLLING_WINDOW, min_periods=1).mean()
    rolling_freeze = combined["freeze_attempted"].astype(float).rolling(ROLLING_WINDOW, min_periods=1).mean()

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(combined["cum_timesteps"], rolling_d, color="steelblue", linewidth=2)
    axes[0].axhline(D_TOL_MM, color="red", linestyle="--", label=f"d_tol={D_TOL_MM}mm")
    axes[0].set_ylabel(f"rolling median terminal d (mm), window={ROLLING_WINDOW}")
    axes[0].set_title("Terminal d vs timesteps")
    axes[0].legend()
    axes[1].plot(combined["cum_timesteps"], rolling_success, color="green", linewidth=2)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel(f"rolling success rate, window={ROLLING_WINDOW}")
    axes[1].set_title("Success rate vs timesteps")
    axes[2].plot(combined["cum_timesteps"], rolling_freeze, color="orange", linewidth=2)
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_xlabel("timesteps")
    axes[2].set_ylabel(f"rolling freeze-attempt fraction, window={ROLLING_WINDOW}")
    axes[2].set_title("Freeze-attempt fraction vs timesteps")
    fig.suptitle(f"Two-lever smoke train -- subtask_max_steps={NEW_SUBTASK_MAX_STEPS}, "
                 f"shaping_mode={NEW_SHAPING_MODE}, ent_coef={SMOKE_ENT_COEF}, "
                 f"{SMOKE_TOTAL_TIMESTEPS} steps\nVERDICT: {verdict}", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "smoke_trajectories.png", dpi=130)
    plt.close(fig)

    summary = dict(
        subtask_max_steps=NEW_SUBTASK_MAX_STEPS, shaping_mode=NEW_SHAPING_MODE,
        ent_coef=SMOKE_ENT_COEF, n_envs=SMOKE_N_ENVS, total_timesteps=SMOKE_TOTAL_TIMESTEPS,
        seed=SMOKE_SEED, n_episodes=n,
        overall_success_rate=overall_success, overall_freeze_attempted_rate=overall_freeze,
        late_success_rate=late_success, late_freeze_attempted_rate=late_freeze,
        late_median_terminal_d_mm=late_d, first_quarter_median_terminal_d_mm=first_quarter_d,
        verdict=verdict,
    )
    with open(OUT_DIR / "smoke_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== BASELINE oracle re-check (subtask_max_steps=60, shaping_mode=additive) ===")
    baseline = run_oracle_gate(subtask_max_steps=60, shaping_mode="additive")

    print(f"\n=== LEVERS APPLIED: oracle gate (subtask_max_steps={NEW_SUBTASK_MAX_STEPS}, "
          f"shaping_mode={NEW_SHAPING_MODE}) ===")
    gated = run_oracle_gate(subtask_max_steps=NEW_SUBTASK_MAX_STEPS, shaping_mode=NEW_SHAPING_MODE)

    print("\n=== Reward readout at settling point (alpha=0.30deg, d=35mm) ===")
    ro_additive = reward_readout("additive")
    ro_mult = reward_readout("multiplicative")
    print(f"  additive:       total={ro_additive['total']:.4f}/{ro_additive['total_max']:.4f} "
          f"= {ro_additive['pct_banked']:.1f}% banked")
    print(f"  multiplicative: total={ro_mult['total']:.4f}/{ro_mult['total_max']:.4f} "
          f"= {ro_mult['pct_banked']:.1f}% banked "
          f"(f(alpha)={ro_mult['f_alpha']:.4f}, g(d)={ro_mult['g_d']:.4f})")

    gate_result = dict(baseline=baseline, gated=gated,
                        reward_readout_additive=ro_additive, reward_readout_multiplicative=ro_mult)
    with open(OUT_DIR / "gate_result.json", "w") as f:
        json.dump(gate_result, f, indent=2)

    gate_passed = gated["success_rate"] >= GATE_MIN_SUCCESS_RATE
    print(f"\n=== GATE: {'PASS' if gate_passed else 'FAIL'} "
          f"(success_rate={gated['success_rate']:.4f}, threshold={GATE_MIN_SUCCESS_RATE}) ===")

    if not gate_passed:
        print("\nGate did NOT pass -- STOPPING per hard rule 3. No training will run.")
        print(f"Diagnosis needed: success_rate={gated['success_rate']:.4f} vs baseline "
              f"{baseline['success_rate']:.4f}. See outcome breakdown above for whether this is "
              f"still timeout-bound (budget needs to go higher) or something else.")
        return

    print("\nGate passed -- proceeding to the single 40k-step PPO smoke train.")
    run_smoke_train()


if __name__ == "__main__":
    main()
