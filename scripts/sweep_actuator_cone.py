"""Sweep the actuator tilt-cone half-angle (`actuator_limit_deg`) and
re-measure reachability at each width, to decide the smallest cone that
restores head reachability without exceeding the ACTUATOR_LIMIT_DEG_CAP
(80deg) hard cap.

Reuses `scripts/measure_reachability.py::run()` (unmodified reachability
logic, only now parameterized by cone width) at each width in
CONE_WIDTHS_DEG. Does not touch phantom geometry, features.py, the reward
formulas, or the acquisition tolerances -- only the actuator cone width
varies across runs.

Usage: uv run python scripts/sweep_actuator_cone.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.custom_env import TARGET_SEQUENCE, ACTUATOR_LIMIT_DEG_CAP
from scripts.measure_reachability import run as measure_reachability

OUT_DIR = Path(__file__).resolve().parent.parent / "logs" / "reachability"
CONE_WIDTHS_DEG = [60.0, 70.0, 75.0, 80.0]
N_SAMPLES = 500
HEAD_TARGET_THRESHOLD = 0.85
JOINT_TARGET_THRESHOLD = 0.80


def run(n_samples: int = N_SAMPLES, seed: int = 0):
    assert max(CONE_WIDTHS_DEG) <= ACTUATOR_LIMIT_DEG_CAP, "sweep must not exceed the hard cap"

    results = {}
    for cone in CONE_WIDTHS_DEG:
        print(f"\n=== actuator_limit_deg = {cone} ===")
        report = measure_reachability(n_samples=n_samples, seed=seed, actuator_limit_deg=cone,
                                       out_dir=OUT_DIR)
        results[cone] = report

    print("\n\n=== CONE-WIDTH x TARGET REACHABILITY TABLE ===")
    header = f"{'cone(deg)':>10s} | " + " | ".join(f"{t:>8s}" for t in TARGET_SEQUENCE) + " | " + f"{'JOINT':>8s}"
    print(header)
    print("-" * len(header))
    for cone in CONE_WIDTHS_DEG:
        r = results[cone]
        row = f"{cone:>10.0f} | " + " | ".join(
            f"{r['per_target_reachable_fraction'][t]:>8.3f}" for t in TARGET_SEQUENCE
        ) + " | " + f"{r['all_three_reachable_fraction']:>8.3f}"
        print(row)

    # Pick the smallest cone meeting BOTH thresholds
    recommended = None
    for cone in CONE_WIDTHS_DEG:
        r = results[cone]
        head_ok = r["per_target_reachable_fraction"]["head"] >= HEAD_TARGET_THRESHOLD
        joint_ok = r["all_three_reachable_fraction"] >= JOINT_TARGET_THRESHOLD
        if head_ok and joint_ok:
            recommended = cone
            break

    print()
    if recommended is not None:
        r = results[recommended]
        print(f"RECOMMENDATION: actuator_limit_deg = {recommended} "
              f"(head={r['per_target_reachable_fraction']['head']:.3f} >= {HEAD_TARGET_THRESHOLD}, "
              f"joint={r['all_three_reachable_fraction']:.3f} >= {JOINT_TARGET_THRESHOLD})")
    else:
        max_cone = max(CONE_WIDTHS_DEG)
        r = results[max_cone]
        print(f"NO cone width up to the {ACTUATOR_LIMIT_DEG_CAP}deg hard cap reaches the "
              f"target thresholds (head>={HEAD_TARGET_THRESHOLD}, joint>={JOINT_TARGET_THRESHOLD}). "
              f"At {max_cone}deg: head={r['per_target_reachable_fraction']['head']:.3f}, "
              f"joint={r['all_three_reachable_fraction']:.3f}. STOPPING -- do not widen further "
              f"or touch d_tol; this is a human decision now (see the d-at-alpha-min backup "
              f"number in logs/head_reachability_diagnosis/).")

    summary = dict(
        cone_widths_deg=CONE_WIDTHS_DEG,
        n_samples=n_samples,
        results={str(c): results[c] for c in CONE_WIDTHS_DEG},
        recommended_actuator_limit_deg=recommended,
        head_threshold=HEAD_TARGET_THRESHOLD,
        joint_threshold=JOINT_TARGET_THRESHOLD,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "cone_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved sweep summary to {OUT_DIR / 'cone_sweep_summary.json'}")
    return summary


if __name__ == "__main__":
    run()
