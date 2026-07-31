"""Report-ready outputs for the locked-environment 4-algorithm grid launch
(status.md "lock the environment + launch the grids" pass): hyperparameter
tables, cumulative reward curves, DQN loss curve, PG entropy curves, and a
convergence plot -- all generated from REAL per-run logs
(logs/<algo>/<run_id>/monitor*.csv + progress.csv), no synthetic data.

Called by scripts/launch_grids.py after the grid runs complete, using the
run metadata `training.sweep.run_sweep` returns (run_id/combo/seed/score/
log_dir) directly -- not a brittle post-hoc glob.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO_ROOT / "logs" / "plots"
TABLES_DIR = REPO_ROOT / "logs" / "tables"

ROLLING_WINDOW = 20
TAIL_FRACTION = 0.2  # "final performance" = mean over the last 20% of episodes


def _load_monitor(log_dir: str) -> pd.DataFrame | None:
    frames = []
    for p in sorted(Path(log_dir).glob("monitor*.csv")):
        try:
            df = pd.read_csv(p, skiprows=1)
        except Exception:
            continue
        if "r" in df.columns and len(df) > 0:
            frames.append(df)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True).sort_values("t").reset_index(drop=True)
    combined["cum_timesteps"] = combined["l"].cumsum()
    return combined


# Sample-efficiency threshold. Radius-40 (the grid sweep's differentiation-
# checked start distance, see status.md) makes essentially every reasonable
# config converge to a ~85% final-success ceiling -- a "timesteps-to-90%"
# metric would return None for most runs (never quite gets there in a lean
# grid budget) and be useless for ranking. 70% is comfortably below that
# ceiling so it's actually reached, at different speeds, by different
# hyperparameter combos -- that speed IS the differentiation signal this
# pass needs, not final success itself.
SUCCESS_TARGET = 0.7


def _final_stats(df: pd.DataFrame) -> dict:
    n = len(df)
    tail = df.iloc[int(n * (1 - TAIL_FRACTION)):]
    # Stability/variance is computed over the WHOLE run (not just the tail)
    # -- at a lean grid budget many runs never reach a long settled tail,
    # so a whole-run volatility measure is the more robust "how stable is
    # this config" signal, and a genuinely separating one: an unstable
    # config swings a lot over its whole trajectory even if its tail
    # happens to look calm.
    reward_std = float(df["r"].std()) if n > 1 else 0.0
    reward_mean_abs = float(df["r"].abs().mean()) if n > 0 else 0.0
    stats = dict(
        n_episodes=n,
        reward_std=reward_std,
        reward_cv=(reward_std / reward_mean_abs) if reward_mean_abs > 1e-9 else None,
        final_mean_reward=float(tail["r"].mean()),
    )
    if "success" in df.columns:
        stats["final_success_rate"] = float(tail["success"].astype(float).mean())
        rolling_success = df["success"].astype(float).rolling(ROLLING_WINDOW, min_periods=1).mean()
        # Sample efficiency: first timestep at which the rolling success
        # rate reaches SUCCESS_TARGET and HOLDS at or above it for the rest
        # of the run (same "no fluke" check used elsewhere in this
        # project's oracle/calibration scripts) -- None if never reached.
        reach_idx = None
        for i in range(n):
            if rolling_success.iloc[i:].min() >= SUCCESS_TARGET:
                reach_idx = i
                break
        stats[f"timesteps_to_{int(SUCCESS_TARGET*100)}pct_success"] = (
            int(df["cum_timesteps"].iloc[reach_idx]) if reach_idx is not None else None
        )
        # Area under the success-vs-timesteps curve, normalized by total
        # duration -- summarizes how much of the run was actually spent
        # succeeding (reward early + sustained success beats late/brief
        # success), independent of the timesteps-to-X% cliff metric above.
        total_t = df["cum_timesteps"].iloc[-1]
        stats["success_auc"] = float(np.trapz(rolling_success, df["cum_timesteps"]) / total_t) if total_t > 0 else 0.0
    if "freeze_attempted" in df.columns:
        stats["final_freeze_attempted_rate"] = float(tail["freeze_attempted"].astype(float).mean())
    return stats


# Column order for the rendered/saved tables: hyperparameters first (added
# per-row from `combo`), then LEAD with sample-efficiency + stability
# (the metrics that actually separate configs at this budget/radius),
# final performance SECOND as supporting context, not the headline.
HEADLINE_METRIC_ORDER = [
    f"timesteps_to_{int(SUCCESS_TARGET*100)}pct_success", "success_auc",
    "reward_std", "reward_cv",
    "final_success_rate", "final_mean_reward", "final_freeze_attempted_rate",
    "n_episodes",
]


def build_hyperparam_table(algo: str, results: list[dict]) -> pd.DataFrame:
    """One row per completed run: hyperparameters (from its combo) as
    columns, THEN sample-efficiency/stability (headline), THEN final
    performance (secondary). Failed runs are not included here (see the
    per-algo failure list in the summary instead)."""
    rows = []
    for r in results:
        df = _load_monitor(r["log_dir"])
        if df is None:
            continue
        stats = _final_stats(df)
        row = dict(run_id=r["run_id"], seed=r["seed"], **r["combo"], **stats)
        rows.append(row)
    table = pd.DataFrame(rows)
    if not table.empty:
        hyperparam_cols = [c for c in table.columns if c not in HEADLINE_METRIC_ORDER
                            and c not in ("run_id", "seed")]
        ordered = (["run_id", "seed"] + hyperparam_cols
                   + [c for c in HEADLINE_METRIC_ORDER if c in table.columns])
        table = table[ordered]
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / f"{algo}_hyperparameter_table.csv", index=False)
    return table


def render_table_png(algo: str, table: pd.DataFrame):
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(min(2 + len(table.columns) * 1.4, 22), 1 + 0.4 * len(table)))
    ax.axis("off")
    display_table = table.copy()
    for col in display_table.select_dtypes(include=[float]).columns:
        display_table[col] = display_table[col].round(4)
    tbl = ax.table(cellText=display_table.values, colLabels=display_table.columns,
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    ax.set_title(f"{algo.upper()} hyperparameter grid (N={len(table)} runs, single_target, locked env)",
                 fontsize=11, pad=20)
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / f"{algo}_table.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_reward_curves(all_results: dict[str, list[dict]]):
    algos = list(all_results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, algo in zip(axes.flat, algos):
        for r in all_results[algo]:
            df = _load_monitor(r["log_dir"])
            if df is None:
                continue
            rolling_r = df["r"].rolling(ROLLING_WINDOW, min_periods=1).mean()
            label = "+".join(f"{k}={v}" for k, v in list(r["combo"].items())[:2]) or "default"
            ax.plot(df["cum_timesteps"], rolling_r, alpha=0.7, linewidth=1.2, label=label)
        ax.set_title(f"{algo.upper()}")
        ax.set_xlabel("timesteps")
        ax.set_ylabel(f"rolling mean reward, window={ROLLING_WINDOW}")
        ax.legend(fontsize=6, loc="lower right")
    fig.suptitle("Cumulative reward curves -- all four algorithms (single_target, locked env)", fontsize=13)
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "cumulative_reward_curves.png", dpi=130)
    plt.close(fig)


def plot_dqn_loss_curve(results: list[dict]):
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for r in results:
        prog_path = Path(r["log_dir"]) / "progress.csv"
        if not prog_path.exists():
            continue
        try:
            df = pd.read_csv(prog_path)
        except Exception:
            continue
        loss_col = next((c for c in df.columns if c.endswith("loss") and "train" in c), None)
        step_col = next((c for c in df.columns if "timesteps" in c), None)
        if loss_col is None or step_col is None or df[loss_col].isna().all():
            continue
        label = "+".join(f"{k}={v}" for k, v in list(r["combo"].items())[:2]) or "default"
        ax.plot(df[step_col], df[loss_col], alpha=0.7, linewidth=1.2, label=label)
        plotted = True
    if plotted:
        ax.set_xlabel("timesteps")
        ax.set_ylabel("DQN train loss")
        ax.set_title("DQN training loss vs timesteps (single_target, locked env)")
        ax.legend(fontsize=6)
    else:
        ax.text(0.5, 0.5, "No DQN loss data found in progress.csv", ha="center", va="center")
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "dqn_loss_curve.png", dpi=130)
    plt.close(fig)
    return plotted


def plot_pg_entropy_curves(all_results: dict[str, list[dict]]):
    """PG entropy for A2C/PPO (SB3 logs 'train/entropy_loss') and REINFORCE
    (its own progress.csv has a direct 'entropy' column, see
    training/reinforce.py::learn)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    specs = [("a2c", "train/entropy_loss", "time/total_timesteps"),
             ("ppo", "train/entropy_loss", "time/total_timesteps"),
             ("reinforce", "entropy", "timesteps")]
    for ax, (algo, ent_col, step_col) in zip(axes, specs):
        plotted = False
        for r in all_results.get(algo, []):
            prog_path = Path(r["log_dir"]) / "progress.csv"
            if not prog_path.exists():
                continue
            try:
                df = pd.read_csv(prog_path)
            except Exception:
                continue
            if ent_col not in df.columns or step_col not in df.columns:
                continue
            label = "+".join(f"{k}={v}" for k, v in list(r["combo"].items())[:2]) or "default"
            ax.plot(df[step_col], df[ent_col], alpha=0.7, linewidth=1.2, label=label)
            plotted = True
        ax.set_title(f"{algo.upper()} entropy")
        ax.set_xlabel("timesteps")
        if not plotted:
            ax.text(0.5, 0.5, "no entropy data found", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.legend(fontsize=6)
    fig.suptitle("Policy-gradient entropy curves vs timesteps", fontsize=13)
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "pg_entropy_curves.png", dpi=130)
    plt.close(fig)


def plot_convergence(all_results: dict[str, list[dict]]):
    """Best-config (by final_mean_reward) success-rate-vs-timesteps curve
    per algorithm, overlaid -- the cross-algorithm convergence comparison."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for algo, results in all_results.items():
        best_score, best_df = -np.inf, None
        for r in results:
            df = _load_monitor(r["log_dir"])
            if df is None or "success" not in df.columns:
                continue
            stats = _final_stats(df)
            if stats["final_mean_reward"] > best_score:
                best_score, best_df = stats["final_mean_reward"], df
        if best_df is None:
            continue
        rolling_success = best_df["success"].astype(float).rolling(ROLLING_WINDOW, min_periods=1).mean()
        ax.plot(best_df["cum_timesteps"], rolling_success, linewidth=1.8, label=algo.upper())
    ax.set_xlabel("timesteps")
    ax.set_ylabel(f"rolling success rate, window={ROLLING_WINDOW}")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Convergence: best-config success rate vs timesteps, all four algorithms")
    ax.legend()
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "convergence_plot.png", dpi=130)
    plt.close(fig)


def write_algo_summary(algo: str, results: list[dict], failures: list[dict]) -> dict:
    best = None
    for r in results:
        df = _load_monitor(r["log_dir"])
        if df is None:
            continue
        stats = _final_stats(df)
        if best is None or stats["final_mean_reward"] > best["final_mean_reward"]:
            best = dict(run_id=r["run_id"], combo=r["combo"], seed=r["seed"], **stats)
    summary = dict(
        algo=algo, n_completed=len(results), n_failed=len(failures),
        failures=failures, best=best,
    )
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLES_DIR / f"{algo}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def generate_all(all_results: dict[str, list[dict]], all_failures: dict[str, list[dict]]):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for algo, results in all_results.items():
        table = build_hyperparam_table(algo, results)
        render_table_png(algo, table)
        summaries[algo] = write_algo_summary(algo, results, all_failures.get(algo, []))

    plot_cumulative_reward_curves(all_results)
    plot_dqn_loss_curve(all_results.get("dqn", []))
    plot_pg_entropy_curves(all_results)
    plot_convergence(all_results)

    return summaries
