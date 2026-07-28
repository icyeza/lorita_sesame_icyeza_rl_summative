"""Measure the fraction of randomly-sampled (unfiltered) fetal poses for
which each target plane -- and all three jointly -- are reachable within
the probe actuator's real +-60deg roll/pitch/yaw limits and the env's own
acquisition tolerances.

This replaces the earlier hand-waved "1-2 of 8 phantoms" estimate from
`scripts/validate_reward_field.py` with a real measured number over a much
larger sample, and is what `UltrasoundProbeEnv(guarantee_reachable=True)`
resamples against in `reset()`.

Uses `UltrasoundProbeEnv._is_target_reachable` directly -- the exact same
reachability search the env itself uses -- via a `guarantee_reachable=False`
env instance (so `reset()` doesn't itself filter what we're trying to
measure).

Usage: uv run python scripts/measure_reachability.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.custom_env import UltrasoundProbeEnv, TARGET_SEQUENCE

OUT_DIR = Path(__file__).resolve().parent.parent / "logs" / "reachability"
N_SAMPLES = 500
LOW_REACHABILITY_WARN_THRESHOLD = 0.6


def run(n_samples: int = N_SAMPLES, seed: int = 0, actuator_limit_deg: float = 60.0,
        out_dir: Path | None = None, save: bool = True):
    env = UltrasoundProbeEnv(seed=seed, guarantee_reachable=False, actuator_limit_deg=actuator_limit_deg)

    per_target_reachable = {t: 0 for t in TARGET_SEQUENCE}
    all_three_reachable = 0

    for i in range(n_samples):
        env.reset(seed=seed + i)  # unfiltered: guarantee_reachable=False
        results = {t: env._is_target_reachable(t) for t in TARGET_SEQUENCE}
        for t, ok in results.items():
            if ok:
                per_target_reachable[t] += 1
        if all(results.values()):
            all_three_reachable += 1

        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{n_samples} sampled (actuator_limit_deg={actuator_limit_deg})")

    per_target_fraction = {t: per_target_reachable[t] / n_samples for t in TARGET_SEQUENCE}
    overall_fraction = all_three_reachable / n_samples

    report = dict(
        n_samples=n_samples,
        actuator_limit_deg=actuator_limit_deg,
        alpha_tol_deg=float(__import__("numpy").degrees(env.alpha_tol)),
        d_tol_mm=env.d_tol * 1000.0,
        per_target_reachable_fraction=per_target_fraction,
        all_three_reachable_fraction=overall_fraction,
    )

    print(f"\nReachability measurement (unfiltered sampling, N={n_samples}, "
          f"actuator_limit_deg={actuator_limit_deg}):")
    for t in TARGET_SEQUENCE:
        print(f"  {t:10s}: {per_target_fraction[t]:.3f}")
    print(f"  {'ALL THREE':10s}: {overall_fraction:.3f}")

    if overall_fraction < LOW_REACHABILITY_WARN_THRESHOLD:
        print(f"\n*** WARNING: overall reachable fraction {overall_fraction:.3f} is below "
              f"{LOW_REACHABILITY_WARN_THRESHOLD} ***")
        print(f"This means the +-{actuator_limit_deg:.0f}deg actuator cone and the configured "
              f"alpha_tol/d_tol are mismatched enough that resample-until-reachable would "
              f"resample-storm most episodes. This is a DESIGN DECISION for the human (widen "
              f"the actuator cone further, or widen tolerances) -- not something to paper over "
              f"with a higher resample cap.")

    if save:
        target_dir = out_dir or OUT_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_cone{int(actuator_limit_deg)}"
        with open(target_dir / f"report{suffix}.json", "w") as f:
            json.dump(report, f, indent=2)
        with open(target_dir / f"report{suffix}.txt", "w") as f:
            f.write(f"Reachability measurement (unfiltered sampling, N={n_samples}, "
                    f"actuator_limit_deg={actuator_limit_deg})\n")
            f.write(f"alpha_tol={report['alpha_tol_deg']:.1f}deg  d_tol={report['d_tol_mm']:.1f}mm\n")
            for t in TARGET_SEQUENCE:
                f.write(f"{t}: {per_target_fraction[t]:.3f}\n")
            f.write(f"all_three: {overall_fraction:.3f}\n")
        print(f"\nSaved report to {target_dir}")

    return report


if __name__ == "__main__":
    run()
