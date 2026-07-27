"""Generate comparison plots from REAL logs under `logs/<algo>/<run_id>/`.

Nothing here fabricates data -- every curve is read from a `monitor.csv`
(Stable-Baselines3 `Monitor` wrapper / our own REINFORCE writer) or a
`progress.csv` (SB3 logger / REINFORCE logger) produced by an actual
training run. If the only runs available are `--smoke` runs, the resulting
plot titles are annotated "[smoke-test run]" so nobody mistakes a
pipeline-validation curve for a real trained result.

Usage: uv run python -m evaluation.plots
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(REPO_ROOT, "logs")
OUT_DIR = os.path.join(LOGS_DIR, "plots")
ALGOS = ["dqn", "reinforce", "a2c", "ppo"]


def _list_runs(algo: str) -> list[str]:
    pattern = os.path.join(LOGS_DIR, algo, "*")
    return sorted([p for p in glob.glob(pattern) if os.path.isdir(p) and os.path.basename(p) != "best"])


def _is_smoke_run(run_dir: str) -> bool:
    cfg_path = os.path.join(run_dir, "run_config.json")
    if not os.path.exists(cfg_path):
        return False
    with open(cfg_path) as f:
        cfg = json.load(f)
    return cfg.get("total_timesteps", 0) <= 5000 or "smoke" in os.path.basename(run_dir)


def _load_monitor(run_dir: str) -> pd.DataFrame | None:
    path = os.path.join(run_dir, "monitor.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, skiprows=1)
    except Exception:
        return None
    if "r" not in df.columns:
        return None
    df["cum_timesteps"] = df["l"].cumsum()
    return df


def _load_progress(run_dir: str) -> pd.DataFrame | None:
    path = os.path.join(run_dir, "progress.csv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def plot_reward_curves():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    any_data = False
    for ax, algo in zip(axes.flatten(), ALGOS):
        runs = _list_runs(algo)
        if not runs:
            ax.set_title(f"{algo.upper()}: no logs found")
            ax.axis("off")
            continue
        run_dir = runs[-1]  # most recent run
        df = _load_monitor(run_dir)
        title_suffix = " [smoke-test run]" if _is_smoke_run(run_dir) else ""
        if df is None or len(df) == 0:
            ax.set_title(f"{algo.upper()}: no episodes logged{title_suffix}")
            ax.axis("off")
            continue
        any_data = True
        ax.plot(df["cum_timesteps"], df["r"], alpha=0.4, label="episode reward")
        if len(df) >= 3:
            window = min(10, len(df))
            ax.plot(df["cum_timesteps"], df["r"].rolling(window, min_periods=1).mean(),
                    label=f"rolling mean ({window})")
        ax.set_xlabel("timesteps")
        ax.set_ylabel("episode reward")
        ax.set_title(f"{algo.upper()}{title_suffix} (run: {os.path.basename(run_dir)})")
        ax.legend(fontsize=8)

    fig.suptitle("Cumulative reward curves per algorithm (from real logs/)")
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "reward_curves.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path, any_data


def plot_dqn_loss():
    runs = _list_runs("dqn")
    if not runs:
        return None
    run_dir = runs[-1]
    df = _load_progress(run_dir)
    loss_col = next((c for c in (df.columns if df is not None else []) if "loss" in c.lower()), None)
    if df is None or loss_col is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df.get("time/total_timesteps", range(len(df))), df[loss_col])
    title_suffix = " [smoke-test run]" if _is_smoke_run(run_dir) else ""
    ax.set_title(f"DQN loss{title_suffix} (run: {os.path.basename(run_dir)})")
    ax.set_xlabel("timesteps")
    ax.set_ylabel(loss_col)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "dqn_loss.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_pg_entropy():
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for algo in ["reinforce", "a2c", "ppo"]:
        runs = _list_runs(algo)
        if not runs:
            continue
        run_dir = runs[-1]
        df = _load_progress(run_dir)
        if df is None:
            continue
        ent_col = next((c for c in df.columns if "entropy" in c.lower()), None)
        if ent_col is None:
            continue
        x = df.get("time/total_timesteps", df.get("timesteps", range(len(df))))
        ax.plot(x, df[ent_col], label=algo.upper())
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("timesteps")
    ax.set_ylabel("entropy")
    ax.set_title("Policy entropy (from real logs/, smoke runs if that's all that exists)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "pg_entropy.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_convergence():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False
    for algo in ALGOS:
        runs = _list_runs(algo)
        if not runs:
            continue
        run_dir = runs[-1]
        df = _load_monitor(run_dir)
        if df is None or len(df) == 0:
            continue
        window = min(10, len(df))
        ax.plot(df["cum_timesteps"], df["r"].rolling(window, min_periods=1).mean(), label=algo.upper())
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("timesteps")
    ax.set_ylabel("rolling mean episode reward")
    ax.set_title("Convergence comparison across algorithms (real logs/)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "convergence.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_generalization(results_by_algo: dict):
    """results_by_algo: {algo: {"in_distribution": {...}, "held_out_transverse_severe_iugr": {...}}}
    as produced by `evaluation.generalization.run`. Purely a plotting helper
    -- it never invents numbers, only visualizes a dict already computed
    from real eval episodes."""
    algos = list(results_by_algo.keys())
    if not algos:
        return None
    metrics = ["success_rate", "classification_rate"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    width = 0.35
    x = np.arange(len(algos))
    for ax, metric in zip(axes, metrics):
        in_vals = [results_by_algo[a]["in_distribution"].get(metric, 0) or 0 for a in algos]
        ho_vals = [results_by_algo[a]["held_out_transverse_severe_iugr"].get(metric, 0) or 0 for a in algos]
        ax.bar(x - width / 2, in_vals, width, label="in-distribution")
        ax.bar(x + width / 2, ho_vals, width, label="held-out (transverse + severe IUGR)")
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in algos])
        ax.set_title(metric)
        ax.legend(fontsize=8)
    fig.suptitle("Generalization comparison (from real eval episodes)")
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "generalization.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    r1, had_data = plot_reward_curves()
    r2 = plot_dqn_loss()
    r3 = plot_pg_entropy()
    r4 = plot_convergence()
    print("Generated (from real logs/, may be smoke-test only):")
    for r in [r1, r2, r3, r4]:
        if r:
            print(f"  {r}")
    if not had_data:
        print("No training runs found under logs/ yet -- run a --smoke sweep first, "
              "e.g. `uv run python -m training.sweep --algo ppo --smoke`.")


if __name__ == "__main__":
    main()
