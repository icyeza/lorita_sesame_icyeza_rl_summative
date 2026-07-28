"""Per-metric spread check for a completed algorithm's hyperparameter grid
-- used as a mandatory DQN-first go/no-go gate before committing hours to
A2C/PPO/REINFORCE at the same budget (status.md "grid launch with
differentiation guardrails" pass).

At the differentiation-checked mid start radius (40), final success
converges near ~85% for most reasonable configs -- so "are the rows
different" must be judged on sample-efficiency/stability metrics
(timesteps-to-70pct-success, success_auc, reward_std), not final success.

Criterion (concrete, not eyeballed): DIFFERENTIATED if EITHER
  - timesteps_to_70pct_success has >=3 distinct non-None values spanning
    a range >= 20% of the run's total_timesteps budbudget, OR
  - success_auc's coefficient of variation (std/mean) across the combos
    exceeds 0.05 (5%)
Otherwise NOT DIFFERENTIATED -- the budget is too short for these
hyperparameters to bite, and the whole batch should be re-run at a longer
budget instead of continuing to burn hours on near-clone tables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SPREAD_FRACTION_THRESHOLD = 0.20
AUC_CV_THRESHOLD = 0.05
MIN_DISTINCT_VALUES = 3


def check_differentiation(table: pd.DataFrame, total_timesteps_budget: int) -> dict:
    metric_col = next((c for c in table.columns if c.startswith("timesteps_to_")), None)
    report = dict(metric_col=metric_col)

    if metric_col is not None:
        vals = table[metric_col].dropna().tolist()
        distinct = sorted(set(vals))
        spread = (max(distinct) - min(distinct)) if len(distinct) >= 2 else 0
        spread_frac = spread / total_timesteps_budget if total_timesteps_budget > 0 else 0.0
        report["timesteps_to_target_values"] = vals
        report["timesteps_to_target_distinct_count"] = len(distinct)
        report["timesteps_to_target_spread"] = spread
        report["timesteps_to_target_spread_fraction"] = spread_frac
        report["timesteps_to_target_differentiated"] = (
            len(distinct) >= MIN_DISTINCT_VALUES and spread_frac >= SPREAD_FRACTION_THRESHOLD
        )
    else:
        report["timesteps_to_target_differentiated"] = False

    if "success_auc" in table.columns:
        auc = table["success_auc"].dropna()
        auc_mean = float(auc.mean()) if len(auc) else 0.0
        auc_std = float(auc.std()) if len(auc) > 1 else 0.0
        auc_cv = (auc_std / auc_mean) if auc_mean > 1e-9 else 0.0
        report["success_auc_min"] = float(auc.min()) if len(auc) else None
        report["success_auc_max"] = float(auc.max()) if len(auc) else None
        report["success_auc_mean"] = auc_mean
        report["success_auc_cv"] = auc_cv
        report["success_auc_differentiated"] = auc_cv >= AUC_CV_THRESHOLD
    else:
        report["success_auc_differentiated"] = False

    report["differentiated"] = (
        report["timesteps_to_target_differentiated"] or report["success_auc_differentiated"]
    )

    # Also report raw min/max for every numeric metric, for the human-readable printout.
    numeric_cols = [c for c in table.columns
                    if c not in ("run_id",) and pd.api.types.is_numeric_dtype(table[c])]
    per_metric_range = {}
    for c in numeric_cols:
        col = table[c].dropna()
        if len(col):
            per_metric_range[c] = dict(min=float(col.min()), max=float(col.max()), mean=float(col.mean()))
    report["per_metric_range"] = per_metric_range

    return report


def print_report(algo: str, report: dict):
    print(f"\n=== Differentiation check: {algo.upper()} ===")
    for metric, r in report["per_metric_range"].items():
        print(f"  {metric:<32} min={r['min']:.3f}  max={r['max']:.3f}  mean={r['mean']:.3f}")
    print(f"  timesteps_to_target: {report.get('timesteps_to_target_distinct_count', 0)} distinct values, "
          f"spread_fraction={report.get('timesteps_to_target_spread_fraction', 0):.3f} "
          f"(need >={MIN_DISTINCT_VALUES} distinct AND >={SPREAD_FRACTION_THRESHOLD:.0%} spread)")
    print(f"  success_auc: cv={report.get('success_auc_cv', 0):.3f} (need >={AUC_CV_THRESHOLD:.0%})")
    verdict = "DIFFERENTIATED -- proceed" if report["differentiated"] else "NOT DIFFERENTIATED -- bump budget and re-run"
    print(f"  VERDICT: {verdict}")
