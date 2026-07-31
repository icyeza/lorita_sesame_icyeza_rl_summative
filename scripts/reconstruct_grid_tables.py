"""Recover differentiated, multi-metric hyperparameter tables from the
grid launch's REAL, intact per-episode logs (status.md "verify integrity,
recover tables" pass) -- NO re-running.

Why this exists: `training/sweep.py::run_sweep` never forwarded
`info_keywords` to the trainer functions (now fixed for future runs, see
that module), so none of the 35 completed grid runs logged "success" (or
freeze_attempted/d_m/alpha_deg) in their monitor.csv -- only the base `r`
(reward), `l` (episode length), `t` (wall time) columns SB3's Monitor
always writes. Those are enough to rebuild real, differentiated tables:

  - Reconstructed success PROXY: in single_target mode with the default
    SUBTASK_MAX_STEPS=60 (confirmed unchanged by this pass -- the grid
    sweep only overrode the start radius, not the step budget), an
    episode can only end two ways: (a) a successful freeze at some step
    <60 (the ONLY way `terminated=True` fires before the subtask-timeout
    check, since single_target has exactly one subtask == the whole
    episode), or (b) the subtask timeout firing at exactly step 60. So
    `episode_length < 60` is a clean, deterministic proxy for "this
    episode succeeded" -- not the same as the logged flag, but not a
    fuzzy heuristic either; it follows directly from the environment's
    own single-target step() logic. Labeled RECONSTRUCTED PROXY
    throughout, never presented as the real logged flag.
  - Reward-based sample-efficiency/stability: timesteps-to-positive-
    reward (rolling mean reward crosses 0 and stays there), reward
    std/CV over the whole run, final (last-20%-tail) mean reward,
    area-under-the-reward-curve.

Usage: uv run python scripts/reconstruct_grid_tables.py
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

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"
PLOTS_DIR = REPO_ROOT / "logs" / "plots"
TABLES_DIR = REPO_ROOT / "logs" / "tables"

SUBTASK_MAX_STEPS = 60  # environment default, unchanged by the grid sweep's radius override
ROLLING_WINDOW = 20
TAIL_FRACTION = 0.2
REWARD_TARGET = 0.0  # "timesteps to positive reward" sample-efficiency threshold

# The FINAL, actually-used run batch per algorithm (identified by the
# timestamp prefix `training.sweep.run_sweep` gave that batch -- see
# logs/grid_launch_summary.json / the launch log). DQN has two earlier,
# DISCARDED attempts (a leak-bug canary at 214607, and the un-bumped
# 6000-step attempt at 222837) preceding the real, bumped 15000-step run
# at 231349 -- only the latter is the batch actually reported/used.
# n_combos bumped for DQN (10 -> 12) and REINFORCE (10 -> 11) by the
# status.md "ablations, protocol fix and report figures" addendum, which
# appended single-knob ablation runs into these same run-id namespaces via
# scripts/ablation_extra_combos.py. Combos 0-9 are untouched.
RUN_BATCHES = {
    "dqn":       dict(prefix="20260729_231349_single_target_grid", n_combos=12),
    "a2c":       dict(prefix="20260730_020925_single_target_grid", n_combos=10),
    "ppo":       dict(prefix="20260730_053623_single_target_grid", n_combos=10),
    "reinforce": dict(prefix="20260730_083939_single_target_grid", n_combos=11),
}


def _load_monitor(log_dir: Path) -> pd.DataFrame | None:
    frames = []
    for p in sorted(log_dir.glob("monitor*.csv")):
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


def _load_run_config(log_dir: Path) -> dict:
    p = log_dir / "run_config.json"
    if not p.exists():
        return {}
    with open(p) as f:
        cfg = json.load(f)
    return cfg.get("combo", {})


def compute_stats(df: pd.DataFrame) -> dict:
    n = len(df)
    tail = df.iloc[int(n * (1 - TAIL_FRACTION)):]

    # Reconstructed success proxy (see module docstring).
    success_proxy = (df["l"] < SUBTASK_MAX_STEPS).astype(float)
    rolling_success_proxy = success_proxy.rolling(ROLLING_WINDOW, min_periods=1).mean()

    reach_idx = None
    for i in range(n):
        if rolling_success_proxy.iloc[i:].min() >= 0.7:
            reach_idx = i
            break

    total_t = df["cum_timesteps"].iloc[-1]
    reward_std = float(df["r"].std()) if n > 1 else 0.0
    reward_mean_abs = float(df["r"].abs().mean()) if n > 0 else 0.0

    rolling_reward = df["r"].rolling(ROLLING_WINDOW, min_periods=1).mean()
    reward_reach_idx = None
    for i in range(n):
        if rolling_reward.iloc[i:].min() >= REWARD_TARGET:
            reward_reach_idx = i
            break

    return dict(
        n_episodes=n,
        mean_episode_length=float(df["l"].mean()),
        # -- reconstructed-success-proxy metrics --
        success_proxy_rate_overall=float(success_proxy.mean()),
        success_proxy_rate_final=float(tail["l"].lt(SUBTASK_MAX_STEPS).astype(float).mean()),
        timesteps_to_70pct_success_proxy=(
            int(df["cum_timesteps"].iloc[reach_idx]) if reach_idx is not None else None
        ),
        success_proxy_auc=float(np.trapz(rolling_success_proxy, df["cum_timesteps"]) / total_t) if total_t > 0 else 0.0,
        # -- reward-based metrics --
        timesteps_to_positive_reward=(
            int(df["cum_timesteps"].iloc[reward_reach_idx]) if reward_reach_idx is not None else None
        ),
        reward_auc=float(np.trapz(rolling_reward, df["cum_timesteps"]) / total_t) if total_t > 0 else 0.0,
        reward_std=reward_std,
        reward_cv=(reward_std / reward_mean_abs) if reward_mean_abs > 1e-9 else None,
        final_mean_reward=float(tail["r"].mean()),
    )


HEADLINE_ORDER = [
    "timesteps_to_70pct_success_proxy", "success_proxy_auc", "timesteps_to_positive_reward", "reward_auc",
    "reward_std", "reward_cv",
    "success_proxy_rate_final", "final_mean_reward", "success_proxy_rate_overall",
    "n_episodes", "mean_episode_length", "wall_clock_s",
]


def _wall_clock_s(log_dir: Path) -> float | None:
    """Wall-clock seconds for the run, read from the Monitor `t` column
    (seconds since that worker's t_start). For n_envs>1 runs the workers
    start together and run concurrently, so the max across workers is the
    run's duration. Not a stats column -- it comes from the raw monitor
    files, not compute_stats."""
    best = None
    for p in sorted(log_dir.glob("monitor*.csv")):
        try:
            df = pd.read_csv(p, skiprows=1)
        except Exception:
            continue
        if "t" in df.columns and len(df) > 0:
            v = float(df["t"].max())
            best = v if best is None else max(best, v)
    return best


def build_table(algo: str, batch: dict) -> tuple[pd.DataFrame, dict]:
    rows = []
    dfs = {}
    for i in range(batch["n_combos"]):
        run_id = f"{batch['prefix']}_combo{i}_seed0"
        log_dir = LOGS_DIR / algo / run_id
        df = _load_monitor(log_dir)
        if df is None:
            print(f"  WARNING: no monitor data for {algo}/{run_id}, skipping")
            continue
        combo = _load_run_config(log_dir)
        stats = compute_stats(df)
        stats["wall_clock_s"] = _wall_clock_s(log_dir)
        rows.append(dict(run_id=run_id, **combo, **stats))
        dfs[run_id] = df

    table = pd.DataFrame(rows)
    if not table.empty:
        hyperparam_cols = [c for c in table.columns if c not in HEADLINE_ORDER and c != "run_id"]
        ordered = ["run_id"] + hyperparam_cols + [c for c in HEADLINE_ORDER if c in table.columns]
        table = table[ordered]
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / f"{algo}_hyperparameter_table.csv", index=False)
    return table, dfs


def render_table_png(algo: str, table: pd.DataFrame):
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(min(2 + len(table.columns) * 1.3, 24), 1 + 0.4 * len(table)))
    ax.axis("off")
    disp = table.copy()
    for col in disp.select_dtypes(include=[float]).columns:
        disp[col] = disp[col].round(4)
    tbl = ax.table(cellText=disp.values, colLabels=disp.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.4)
    ax.set_title(f"{algo.upper()} hyperparameter grid (N={len(table)} runs, single_target, "
                 f"radius=40, locked env)\nsuccess = RECONSTRUCTED PROXY (episode_length < "
                 f"{SUBTASK_MAX_STEPS}), not a logged flag -- see script docstring",
                 fontsize=9, pad=20)
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / f"{algo}_table.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_reward_curves(all_dfs: dict[str, dict[str, pd.DataFrame]], all_tables: dict[str, pd.DataFrame]):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, algo in zip(axes.flat, all_dfs.keys()):
        for run_id, df in all_dfs[algo].items():
            rolling_r = df["r"].rolling(ROLLING_WINDOW, min_periods=1).mean()
            combo = all_tables[algo].loc[all_tables[algo]["run_id"] == run_id]
            hp_cols = [c for c in all_tables[algo].columns if c not in HEADLINE_ORDER and c != "run_id"][:2]
            label = "+".join(f"{c}={combo[c].values[0]}" for c in hp_cols) if len(combo) else run_id
            ax.plot(df["cum_timesteps"], rolling_r, alpha=0.7, linewidth=1.1, label=label)
        ax.set_title(f"{algo.upper()}")
        ax.set_xlabel("timesteps")
        ax.set_ylabel(f"rolling mean reward, window={ROLLING_WINDOW}")
        ax.legend(fontsize=6, loc="lower right")
    fig.suptitle("Reward learning curves -- all four algorithms, all combos (single_target, radius=40, locked env)",
                 fontsize=13)
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "cumulative_reward_curves.png", dpi=130)
    plt.close(fig)


def plot_dqn_config_spread(dqn_dfs: dict[str, pd.DataFrame], dqn_table: pd.DataFrame):
    """Per-config spread glance for the DQN re-check: reward trajectory for
    EVERY DQN combo overlaid, to see by eye whether any config is climbing
    vs all flat/negative."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for run_id, df in dqn_dfs.items():
        rolling_r = df["r"].rolling(ROLLING_WINDOW, min_periods=1).mean()
        ax.plot(df["cum_timesteps"], rolling_r, alpha=0.8, linewidth=1.3, label=run_id.split("_combo")[-1])
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8, label="reward=0")
    ax.set_xlabel("timesteps")
    ax.set_ylabel(f"rolling mean reward, window={ROLLING_WINDOW}")
    ax.set_title("DQN re-check: reward trajectory, all 10 combos (15000-step bumped run)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "dqn_recheck_reward_spread.png", dpi=130)
    plt.close(fig)


def main():
    all_tables, all_dfs, summaries = {}, {}, {}
    for algo, batch in RUN_BATCHES.items():
        print(f"\n=== {algo.upper()} ===")
        table, dfs = build_table(algo, batch)
        all_tables[algo] = table
        all_dfs[algo] = dfs
        render_table_png(algo, table)
        print(table.to_string(index=False))

        best_idx = table["final_mean_reward"].idxmax() if not table.empty else None
        best = table.loc[best_idx].to_dict() if best_idx is not None else None
        summaries[algo] = dict(n_runs=len(table), best=best)
        with open(TABLES_DIR / f"{algo}_summary.json", "w") as f:
            json.dump(summaries[algo], f, indent=2, default=str)

    plot_reward_curves(all_dfs, all_tables)
    plot_dqn_config_spread(all_dfs["dqn"], all_tables["dqn"])

    print("\n\n=== DQN RE-CHECK (reward-based, not the broken success gate) ===")
    dqn_table = all_tables["dqn"]
    print(f"final_mean_reward across DQN's 10 combos: min={dqn_table['final_mean_reward'].min():.3f}, "
          f"max={dqn_table['final_mean_reward'].max():.3f}, mean={dqn_table['final_mean_reward'].mean():.3f}")
    print(f"success_proxy_rate_final across DQN's 10 combos: min={dqn_table['success_proxy_rate_final'].min():.3f}, "
          f"max={dqn_table['success_proxy_rate_final'].max():.3f}")
    all_negative = (dqn_table["final_mean_reward"] < 0).all()
    print(f"ALL 10 combos have negative final_mean_reward: {all_negative}")

    print(f"\nTables: {TABLES_DIR}")
    print(f"Plots: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
