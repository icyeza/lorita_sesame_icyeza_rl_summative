"""Budget-headroom sweep + timeout-flattening gate (NON-training).

Follow-up to scripts/lever_fix_oracle_gate.py: that pass found Lever 1
(subtask_max_steps 60->150) alone lifted the oracle from 27%->70%, and
Lever 2 (shaping_mode="multiplicative") was independently confirmed via
the reward-field gate to fix the reward's gradient across ALL THREE
targets (head 0.07->0.78, abdomen 0.16->0.92, femur already good). This
pass keeps both and asks a narrower question: how much step headroom makes
a LEARNER (not just an oracle) viable? The gate is not a fixed oracle
success number (the prior pass's 85% was called out as arbitrary) but
whether the TIMEOUT RATE FLATTENS as budget grows -- that property is what
tells us "improving maps to succeeding," which is what RL needs to
bootstrap a gradient.

IMPORTANT STRUCTURAL FINDING (verified before running the sweep, see
status.md): `EPISODE_MAX_STEPS` (180, a separate, NOT-permitted-to-change
constant -- this pass's hard rules only allow changing
`SUBTASK_MAX_STEPS`) silently caps every single-target episode at 180
total steps, because `steps_in_subtask` and `total_steps` are numerically
identical when there is exactly one subtask (single_target=True). Verified
directly: envs constructed with subtask_max_steps=250 and
subtask_max_steps=400 both truncate an unproductive episode at EXACTLY
step 180 (truncated=True via the EPISODE cap, not the subtask cap) --
bit-identical behavior. So while this script runs the full 150/250/400
sweep as specified, 250 and 400 are NOT genuinely different budgets from
each other (or from ~180) for single-target femur episodes -- this is
reported plainly as the primary finding, not fixed by also changing
EPISODE_MAX_STEPS (out of scope for this pass, a human decision).

No RL training happens unless the gate (defined in the module docstring
above) passes. Nothing here changes UltrasoundProbeEnv's defaults.

Usage: uv run python scripts/budget_headroom_gate.py
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

from environment.custom_env import UltrasoundProbeEnv, EPISODE_MAX_STEPS
from scripts.diagnose_scripted_policy import greedy_episode, TARGET
from training.pg_training import train_ppo

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "logs" / "budget_headroom_gate"
SMOKE_LOG_DIR = OUT_DIR / "smoke_train"
SMOKE_MODEL_DIR = REPO_ROOT / "models" / "budget_headroom_gate" / "smoke_train"

SHAPING_MODE = "multiplicative"  # kept from the prior pass, per hard rule 2
BUDGETS = [150, 250, 400]
N_ORACLE_EPISODES = 100
ORACLE_SEED = 42

# Gate thresholds (this pass's own, explicit, not silently re-picked to fit
# the data -- see module docstring / status.md for the reasoning):
GATE_SUCCESS_FLOOR = 0.85          # success at 250 must be high...
GATE_MARGIN_STEP_FRACTION = 0.8    # ...with median steps-to-success <= 80% of the 250 budget
GATE_FLATTEN_ABS_MAX = 0.05        # 250->400 timeout-rate drop must be <=5 percentage points...
GATE_FLATTEN_RATIO_MAX = 0.3       # ...AND <=30% of the 150->250 drop (diminishing returns)

NEAR_D_THRESHOLD_MM = 24.0  # "close, just ran out of clock" vs "never got close" (2x d_tol)


def run_oracle_at_budget(subtask_max_steps: int, n: int = N_ORACLE_EPISODES, seed: int = ORACLE_SEED):
    env = UltrasoundProbeEnv(single_target=True, single_target_which=TARGET, seed=seed,
                              subtask_max_steps=subtask_max_steps, shaping_mode=SHAPING_MODE)
    results = [greedy_episode(env, max_steps=subtask_max_steps + 20) for _ in range(n)]
    n_success = sum(r["success"] for r in results)
    outcomes = {}
    for r in results:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    n_timeout = outcomes.get("timeout_subtask", 0) + outcomes.get("timeout_episode", 0)
    steps_success = [r["steps"] for r in results if r["success"]]

    stats = dict(
        subtask_max_steps=subtask_max_steps, n=n,
        success_rate=n_success / n, timeout_rate=n_timeout / n,
        median_steps_to_success=float(np.median(steps_success)) if steps_success else None,
        outcomes=outcomes, results=results,
    )
    print(f"  subtask_max_steps={subtask_max_steps}: success_rate={stats['success_rate']:.4f} "
          f"({n_success}/{n}), timeout_rate={stats['timeout_rate']:.4f}, "
          f"median steps-to-success={stats['median_steps_to_success']}, outcomes={outcomes}")
    return stats


def terminal_d_split(stats: dict):
    results = stats["results"]
    d_success = np.array([r["terminal_d_mm"] for r in results if r["success"]])
    d_timeout = np.array([r["terminal_d_mm"] for r in results
                           if r["outcome"] in ("timeout_subtask", "timeout_episode")])
    split = dict(
        n_success=len(d_success), n_timeout=len(d_timeout),
        d_success_median=float(np.median(d_success)) if len(d_success) else None,
        d_success_mean=float(np.mean(d_success)) if len(d_success) else None,
        d_timeout_median=float(np.median(d_timeout)) if len(d_timeout) else None,
        d_timeout_mean=float(np.mean(d_timeout)) if len(d_timeout) else None,
        frac_timeout_near=float(np.mean(d_timeout <= NEAR_D_THRESHOLD_MM)) if len(d_timeout) else None,
    )
    print(f"  terminal d | succeeded (N={split['n_success']}): "
          f"median={split['d_success_median']:.2f}mm mean={split['d_success_mean']:.2f}mm")
    if len(d_timeout):
        print(f"  terminal d | timed-out (N={split['n_timeout']}): "
              f"median={split['d_timeout_median']:.2f}mm mean={split['d_timeout_mean']:.2f}mm, "
              f"fraction <= {NEAR_D_THRESHOLD_MM}mm (close, just ran out of clock)="
              f"{split['frac_timeout_near']:.3f}")
    else:
        print("  no timed-out episodes at this budget")
    return split


def run_smoke_train(subtask_max_steps: int):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SMOKE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    SMOKE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(single_target=True, single_target_which=TARGET,
                       subtask_max_steps=subtask_max_steps, shaping_mode=SHAPING_MODE)
    config = dict(entropy_coef=0.05)
    info_keywords = ("success", "freeze_attempted", "d_m", "alpha_deg")

    print(f"\nSmoke train: PPO, femur-only, subtask_max_steps={subtask_max_steps}, "
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

    print(f"\nSmoke train (N={n} episodes):")
    print(f"  Overall: success={overall_success:.4f}, freeze={overall_freeze:.4f}")
    print(f"  First quarter median terminal d: {first_d:.2f}mm")
    print(f"  Last quarter: success={late_success:.4f}, freeze={late_freeze:.4f}, "
          f"median terminal d={late_d:.2f}mm")

    learning_signal = (late_success > 0.0) or (late_d < first_d * 0.9)
    verdict = "LEARNING SIGNAL PRESENT" if learning_signal else "FLAT AT 40k"
    print(f"  VERDICT: {verdict}")

    rolling_d = combined["d_mm"].rolling(50, min_periods=1).median()
    rolling_success = combined["success"].astype(float).rolling(50, min_periods=1).mean()
    rolling_freeze = combined["freeze_attempted"].astype(float).rolling(50, min_periods=1).mean()

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(combined["cum_timesteps"], rolling_d, color="steelblue", linewidth=2)
    axes[0].axhline(12.0, color="red", linestyle="--", label="d_tol=12mm")
    axes[0].set_ylabel("rolling median terminal d (mm), window=50")
    axes[0].set_title("Terminal d vs timesteps")
    axes[0].legend()
    axes[1].plot(combined["cum_timesteps"], rolling_success, color="green", linewidth=2)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("rolling success rate, window=50")
    axes[1].set_title("Success rate vs timesteps")
    axes[2].plot(combined["cum_timesteps"], rolling_freeze, color="orange", linewidth=2)
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_xlabel("timesteps")
    axes[2].set_ylabel("rolling freeze-attempt fraction, window=50")
    axes[2].set_title("Freeze-attempt fraction vs timesteps")
    fig.suptitle(f"Budget-headroom smoke train -- subtask_max_steps={subtask_max_steps}, "
                 f"shaping_mode={SHAPING_MODE}, ent_coef=0.05, 40000 steps\nVERDICT: {verdict}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "smoke_trajectories.png", dpi=130)
    plt.close(fig)

    summary = dict(
        subtask_max_steps=subtask_max_steps, shaping_mode=SHAPING_MODE, ent_coef=0.05,
        n_envs=4, total_timesteps=40_000, seed=0, n_episodes=n,
        overall_success_rate=overall_success, overall_freeze_attempted_rate=overall_freeze,
        late_success_rate=late_success, late_freeze_attempted_rate=late_freeze,
        late_median_terminal_d_mm=late_d, first_quarter_median_terminal_d_mm=first_d,
        verdict=verdict,
    )
    with open(OUT_DIR / "smoke_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"NOTE: EPISODE_MAX_STEPS={EPISODE_MAX_STEPS} is a separate, not-permitted-to-change "
          f"constant that caps every single-target episode's TOTAL steps -- verified that "
          f"subtask_max_steps=250 and 400 both truncate an unproductive episode at exactly step "
          f"{EPISODE_MAX_STEPS} (bit-identical). See module docstring.\n")

    print("=== Oracle sweep (non-training) at subtask_max_steps in {150, 250, 400} ===")
    sweep = {}
    for budget in BUDGETS:
        sweep[budget] = run_oracle_at_budget(budget)

    print("\n=== Terminal-d split (succeeded vs timed-out) at budget=250 ===")
    split_250 = terminal_d_split(sweep[250])

    timeout_150 = sweep[150]["timeout_rate"]
    timeout_250 = sweep[250]["timeout_rate"]
    timeout_400 = sweep[400]["timeout_rate"]
    drop_150_250 = timeout_150 - timeout_250
    drop_250_400 = timeout_250 - timeout_400
    flattening = (drop_250_400 <= GATE_FLATTEN_ABS_MAX) and \
                 (drop_250_400 <= GATE_FLATTEN_RATIO_MAX * max(drop_150_250, 1e-9))

    success_250 = sweep[250]["success_rate"]
    median_steps_250 = sweep[250]["median_steps_to_success"]
    margin_ok = (median_steps_250 is not None) and (median_steps_250 <= GATE_MARGIN_STEP_FRACTION * 250)
    success_ok = success_250 >= GATE_SUCCESS_FLOOR

    print(f"\n=== Timeout-rate table ===")
    print(f"  150: {timeout_150:.4f}   250: {timeout_250:.4f}   400: {timeout_400:.4f}")
    print(f"  drop 150->250: {drop_150_250:.4f}   drop 250->400: {drop_250_400:.4f}")
    print(f"  flattening (250->400 drop <= {GATE_FLATTEN_ABS_MAX} abs AND "
          f"<= {GATE_FLATTEN_RATIO_MAX:.0%} of 150->250 drop): {flattening}")

    gate_passed = success_ok and margin_ok and flattening
    print(f"\n=== GATE: {'PASS' if gate_passed else 'FAIL'} ===")
    print(f"  success_250={success_250:.4f} (need >= {GATE_SUCCESS_FLOOR}): {success_ok}")
    print(f"  median_steps_to_success_250={median_steps_250} "
          f"(need <= {GATE_MARGIN_STEP_FRACTION*250:.0f}): {margin_ok}")
    print(f"  timeout flattening 250->400: {flattening}")

    gate_result = dict(
        sweep={str(b): {k: v for k, v in s.items() if k != "results"} for b, s in sweep.items()},
        terminal_d_split_250=split_250,
        timeout_150=timeout_150, timeout_250=timeout_250, timeout_400=timeout_400,
        drop_150_250=drop_150_250, drop_250_400=drop_250_400, flattening=flattening,
        success_250=success_250, median_steps_to_success_250=median_steps_250,
        gate_passed=gate_passed,
    )
    with open(OUT_DIR / "gate_result.json", "w") as f:
        json.dump(gate_result, f, indent=2)

    if not gate_passed:
        print("\nGate did NOT pass -- STOPPING per hard rule 3. No training will run.")
        if not flattening:
            print("Diagnosis: timeouts still dropping steeply at 400 -- BUT this pass found "
                  "250 and 400 are structurally IDENTICAL for single-target episodes due to "
                  f"the untouched EPISODE_MAX_STEPS={EPISODE_MAX_STEPS} cap (see module "
                  "docstring) -- so 'flattening' here cannot be a genuine geometry signal "
                  "either way; a real test of headroom beyond 180 would require also raising "
                  "EPISODE_MAX_STEPS, out of this pass's permitted-changes scope.")
        if flattening and not (success_ok and margin_ok):
            print("Diagnosis: timeouts flattened but success/margin still insufficient -- per "
                  "the brief, report this as a possible start-distribution problem, not a "
                  "budget one. Do not train.")
        return

    print("\nGate passed -- proceeding to the single 40k-step PPO smoke train at budget=250.")
    run_smoke_train(subtask_max_steps=250)


if __name__ == "__main__":
    main()
