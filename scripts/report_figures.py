"""Report figures, written fresh against the canonical grid batches
(status.md "ablations, protocol fix and report figures" addendum).

REPLACES evaluation/plots.py for report purposes. That module plotted
`_list_runs(algo)[-1]` -- the single most recent run directory per
algorithm -- so it never showed a grid, its legends could not
disambiguate, and it plotted SB3's `train/entropy_loss` unnegated
alongside REINFORCE's true entropy. It is left in place (other passes
reference it) but nothing here imports it.

Outputs (200 dpi) to logs/plots/report/:
  reward_curves.png   4 panels, every grid config per algorithm
  dqn_loss.png        DQN TD loss, all 12 configs
  pg_entropy.png      A2C/PPO/REINFORCE policy entropy, SIGN-CORRECTED
  convergence.png     best-per-algorithm + time-to-competence bars
  generalization.png  success by start condition, Wilson 95% CIs

SIGN CONVENTION (verified in the installed SB3, not assumed):
stable_baselines3/a2c/a2c.py:169 and ppo/ppo.py:252 both compute
`entropy_loss = -th.mean(entropy)` and log THAT under
`train/entropy_loss`. training/reinforce.py:146 logs
`dist.entropy().mean()` directly. So the SB3 series are negated here and
the REINFORCE series is not. ln(12)=2.4849 is the uniform-policy entropy
for this env's Discrete(12) action space (12 entries in
environment.custom_env.ACTIONS).

Usage: uv run python scripts/report_figures.py
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
OUT_DIR = LOGS_DIR / "plots" / "report"
TABLES_DIR = LOGS_DIR / "tables"

ROLLING_WINDOW = 20
SUSTAIN_POINTS = 5          # "sustained" = threshold held for 5 consecutive logged points
COMPETENCE_THRESHOLD = 0.0  # episode return = 0
UNIFORM_ENTROPY = float(np.log(12))
DPI = 200

# Okabe-Ito, colour-blind safe. One colour per ALGORITHM, held constant
# across every figure.
ALGO_COLOUR = {
    "dqn": "#D55E00",        # vermillion
    "reinforce": "#0072B2",  # blue
    "a2c": "#009E73",        # bluish green
    "ppo": "#CC79A7",        # reddish purple
}
ALGO_LABEL = {"dqn": "DQN", "reinforce": "REINFORCE", "a2c": "A2C", "ppo": "PPO"}

# Within-panel styles for the "best three" configs of one algorithm.
BEST_N = 3
BEST_STYLES = [dict(linewidth=1.6, alpha=1.0),
               dict(linewidth=1.3, alpha=0.8, linestyle="--"),
               dict(linewidth=1.1, alpha=0.7, linestyle=":")]

RUN_BATCHES = {
    "dqn":       dict(prefix="20260729_231349_single_target_grid", combos=list(range(12))),
    "a2c":       dict(prefix="20260730_020925_single_target_grid", combos=list(range(10))),
    "ppo":       dict(prefix="20260730_053623_single_target_grid", combos=list(range(10))),
    "reinforce": dict(prefix="20260730_083939_single_target_grid", combos=list(range(11))),
}
DQN_ABLATION_COMBOS = {10, 11}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def run_dir(algo: str, combo: int) -> Path:
    return LOGS_DIR / algo / f"{RUN_BATCHES[algo]['prefix']}_combo{combo}_seed0"


def load_monitor(algo: str, combo: int) -> pd.DataFrame | None:
    """Episode stream. Concatenates monitor*.csv (n_envs=4 runs write one
    per worker) and orders by wall-clock `t`, matching how
    scripts/reconstruct_grid_tables.py builds the tables."""
    d = run_dir(algo, combo)
    frames = []
    for p in sorted(d.glob("monitor*.csv")):
        try:
            df = pd.read_csv(p, skiprows=1)
        except Exception:
            continue
        if "r" in df.columns and len(df) > 0:
            frames.append(df)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True).sort_values("t").reset_index(drop=True)
    out["cum_timesteps"] = out["l"].cumsum()
    return out


def load_progress(algo: str, combo: int) -> pd.DataFrame | None:
    p = run_dir(algo, combo) / "progress.csv"
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def load_combo_config(algo: str, combo: int) -> dict:
    p = run_dir(algo, combo) / "run_config.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f).get("combo", {})


def available_combos(algo: str) -> list[int]:
    return [c for c in RUN_BATCHES[algo]["combos"] if run_dir(algo, c).exists()
            and load_monitor(algo, c) is not None]


# ----------------------------------------------------------------------
# Legend labels: grid row + ONLY the hyperparameters that actually vary
# ----------------------------------------------------------------------
_SHORT = {
    "learning_rate": "lr", "batch_size": "b", "target_update_interval": "tu",
    "buffer_size": "buf", "exploration_fraction": "eps_f", "net_arch": "",
    "use_baseline": "base", "entropy_coef": "ent", "n_steps": "ns",
    "clip_range": "clip", "gae_lambda": "lam", "gamma": "g",
}


def _fmt(key: str, val) -> str:
    if key == "net_arch":
        return f"[{','.join(str(v) for v in val)}]"
    if key == "buffer_size":
        return f"buf={int(val) // 1000}k"
    if key == "learning_rate":
        return f"lr={val:g}"
    if isinstance(val, bool):
        return f"{_SHORT[key]}={'T' if val else 'F'}"
    return f"{_SHORT.get(key, key)}={val:g}" if isinstance(val, float) else f"{_SHORT.get(key, key)}={val}"


def build_labels(algo: str, combos: list[int]) -> dict[int, str]:
    """`#3 lr=0.001 b=64 tu=2000 [128,128]` -- grid row plus every
    hyperparameter that differs across THIS grid. Constant-across-the-grid
    keys (e.g. gamma) are dropped: repeating them is what made the old
    legends useless."""
    cfgs = {c: load_combo_config(algo, c) for c in combos}
    keys = list(next(iter(cfgs.values())) if cfgs else {})
    varying = []
    for k in keys:
        vals = {json.dumps(cfgs[c].get(k), sort_keys=True) for c in combos}
        if len(vals) > 1:
            varying.append(k)
    labels = {}
    for c in combos:
        parts = [_fmt(k, cfgs[c][k]) for k in varying if k in cfgs[c]]
        suffix = " (abl.)" if algo == "dqn" and c in DQN_ABLATION_COMBOS else ""
        labels[c] = f"#{c} " + " ".join(parts) + suffix
    return labels


def rank_by_final_reward(algo: str, combos: list[int]) -> list[int]:
    """Best-first by final_mean_reward, recomputed here the same way
    scripts/reconstruct_grid_tables.py does (mean return over the last 20%
    of episodes) so the ranking matches the tables."""
    scores = {}
    for c in combos:
        df = load_monitor(algo, c)
        tail = df.iloc[int(len(df) * 0.8):]
        scores[c] = float(tail["r"].mean())
    return sorted(combos, key=lambda c: scores[c], reverse=True)


# ----------------------------------------------------------------------
# Figure 2 -- reward curves
# ----------------------------------------------------------------------
def fig2_reward_curves():
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2))
    notes = []
    for ax, algo in zip(axes.flat, ["dqn", "reinforce", "a2c", "ppo"]):
        combos = available_combos(algo)
        labels = build_labels(algo, combos)
        ranked = rank_by_final_reward(algo, combos)
        best, rest = ranked[:BEST_N], ranked[BEST_N:]

        for c in rest:
            df = load_monitor(algo, c)
            ax.plot(df["cum_timesteps"], df["r"].rolling(ROLLING_WINDOW, min_periods=1).mean(),
                    color="0.72", linewidth=0.6, alpha=0.85, zorder=1)
        for style, c in zip(BEST_STYLES, best):
            df = load_monitor(algo, c)
            ax.plot(df["cum_timesteps"], df["r"].rolling(ROLLING_WINDOW, min_periods=1).mean(),
                    color=ALGO_COLOUR[algo], label=labels[c], zorder=3, **style)
        ax.axhline(0.0, color="0.35", linestyle=(0, (4, 3)), linewidth=0.7, zorder=2)
        ax.set_title(ALGO_LABEL[algo], color=ALGO_COLOUR[algo])
        ax.set_xlabel("environment timesteps")
        ax.set_ylabel("episode return\n(rolling mean, window = 20)")
        ax.legend(loc="lower left", framealpha=0.92, handlelength=1.5,
                  fontsize=5.2, borderpad=0.3, labelspacing=0.3)
        notes.append(f"{ALGO_LABEL[algo]}: {len(combos)} configs, "
                     f"best 3 coloured ({', '.join(labels[c] for c in best)})")
    fig.suptitle("Learning curves by algorithm and hyperparameter configuration", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT_DIR / "reward_curves.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out, notes


# ----------------------------------------------------------------------
# Figure 3 -- DQN TD loss
# ----------------------------------------------------------------------
def fig3_dqn_loss():
    combos = available_combos("dqn")
    labels = build_labels("dqn", combos)
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    plotted = []
    for c in combos:
        df = load_progress("dqn", c)
        if df is None or "train/loss" not in df.columns:
            continue
        d = df.dropna(subset=["train/loss"])
        if c in DQN_ABLATION_COMBOS:
            style = dict(color="#000000" if c == 10 else "#0072B2",
                         linewidth=1.5, linestyle="--" if c == 10 else "-.",
                         alpha=1.0, zorder=4)
        else:
            style = dict(color=ALGO_COLOUR["dqn"], linewidth=0.7, alpha=0.45, zorder=2)
        ax.plot(d["time/total_timesteps"], d["train/loss"], label=labels[c], **style)
        plotted.append(c)
    ax.set_xlabel("environment timesteps")
    ax.set_ylabel("TD loss")
    ax.set_title("DQN temporal-difference loss")
    handles, labs = ax.get_legend_handles_labels()
    keep = [i for i, c in enumerate(plotted) if c in DQN_ABLATION_COMBOS]
    ax.legend([handles[i] for i in keep], [labs[i] for i in keep],
              loc="upper right", framealpha=0.9, handlelength=2.0, fontsize=5.2,
              title=f"ablations (other {len(plotted) - len(keep)} configs in orange)",
              title_fontsize=5.2)
    fig.tight_layout()
    out = OUT_DIR / "dqn_loss.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out, plotted


# ----------------------------------------------------------------------
# Figure 4 -- policy entropy (sign-corrected)
# ----------------------------------------------------------------------
def fig4_pg_entropy():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
    overlaps = []
    for ax, algo in zip(axes, ["a2c", "ppo", "reinforce"]):
        combos = available_combos(algo)
        labels = build_labels(algo, combos)
        ranked = rank_by_final_reward(algo, combos)
        best, rest = ranked[:BEST_N], ranked[BEST_N:]

        series = {}
        for c in combos:
            df = load_progress(algo, c)
            if df is None:
                continue
            if algo == "reinforce":
                if "entropy" not in df.columns:
                    continue
                x, y = df["timesteps"], df["entropy"]
            else:
                if "train/entropy_loss" not in df.columns:
                    continue
                d = df.dropna(subset=["train/entropy_loss"])
                # SIGN FIX: SB3 logs -mean(entropy); negate for true entropy.
                x, y = d["time/total_timesteps"], -d["train/entropy_loss"]
            series[c] = (np.asarray(x), np.asarray(y))

        # REINFORCE: the entropy_coef-only pairs land on top of one another
        # and would silently hide a run. Overlay the second member so the
        # coincidence is visible.
        dup_of = {}
        if algo == "reinforce":
            keys = list(series)
            for i, a in enumerate(keys):
                for b in keys[i + 1:]:
                    ya, yb = series[a][1], series[b][1]
                    # NOT an exact-equality test. The monitor r/l columns of
                    # these pairs ARE bit-identical (status.md Addendum 19),
                    # but the LOGGED ENTROPY differs by ~1e-5 to 7e-5: the
                    # entropy_coef term does perturb the weights slightly,
                    # just never enough to change a sampled action. So the
                    # curves are visually coincident, not identical, and an
                    # atol=0 test finds nothing.
                    if len(ya) == len(yb) and np.max(np.abs(ya - yb)) < 1e-3:
                        dup_of.setdefault(b, a)
            overlaps = [(v, k) for k, v in dup_of.items()]

        for c in rest:
            if c not in series:
                continue
            x, y = series[c]
            ax.plot(x, y, color="0.72", linewidth=0.6, alpha=0.85, zorder=1)
        for style, c in zip(BEST_STYLES, best):
            if c not in series:
                continue
            x, y = series[c]
            ax.plot(x, y, color=ALGO_COLOUR[algo], label=labels[c], zorder=3, **style)
        for b, a in dup_of.items():
            x, y = series[b]
            lab = None
            if b == min(dup_of):
                lab = (", ".join(f"#{y_} ~ #{x_}" for y_, x_ in sorted(dup_of.items()))
                       + "  (coincident, max diff < 1e-4)")
            ax.plot(x, y, color="#000000", linewidth=0.9, linestyle=(0, (2, 2)),
                    alpha=0.85, zorder=4, label=lab)

        ax.axhline(UNIFORM_ENTROPY, color="0.25", linestyle=(0, (5, 3)), linewidth=0.8, zorder=2)
        ax.annotate("uniform policy", xy=(0.98, UNIFORM_ENTROPY), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=6, color="0.25")
        ax.set_title(ALGO_LABEL[algo], color=ALGO_COLOUR[algo])
        ax.set_xlabel("environment timesteps")
        ax.legend(loc="lower left", framealpha=0.9, handlelength=1.5, fontsize=4.8,
                  borderpad=0.3, labelspacing=0.3)
    axes[0].set_ylabel("policy entropy (nats)")
    axes[0].set_ylim(0, UNIFORM_ENTROPY * 1.08)
    fig.suptitle("Policy entropy during training (Discrete(12) action space)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT_DIR / "pg_entropy.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out, overlaps


# ----------------------------------------------------------------------
# Figure 5 -- convergence
# ----------------------------------------------------------------------
def first_sustained_crossing(df: pd.DataFrame, mode: str = "holds") -> tuple[int | None, pd.Series]:
    """Timesteps to competence (rolling-mean episode return >= 0).

    TWO DEFINITIONS, because the obvious one does not work in this
    environment:

    mode="window5" -- the literal "held for 5 consecutive logged points"
      definition (a logged point = one episode). USELESS HERE: under the
      start curriculum, early episodes begin near the goal and are short
      and easy, so every algorithm's rolling mean is ALREADY well above 0
      at episode 20 (DQN 2.26, A2C 4.96, PPO 4.74, REINFORCE 0.77) and
      then DEGRADES as the replay/rollout distribution fills with harder
      states. All four "cross" within the first few hundred timesteps,
      including DQN, which never becomes competent. The metric measures
      the curriculum's easy start, not learning.

    mode="holds" (default, used for the figure) -- first timestep from
      which the rolling mean stays >= 0 for the REST OF THE RUN. Same rule
      scripts/reconstruct_grid_tables.py uses for its
      `timesteps_to_positive_reward` column, so figure and tables agree.
      Returns None if never reached.

    Both use min_periods=ROLLING_WINDOW so a partly-filled window can
    never trigger a crossing (with min_periods=1 the first episode alone
    can satisfy it)."""
    roll = df["r"].rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()
    vals = roll.to_numpy()
    if mode == "window5":
        for i in range(len(vals) - SUSTAIN_POINTS + 1):
            w = vals[i:i + SUSTAIN_POINTS]
            if not np.isnan(w).any() and np.all(w >= COMPETENCE_THRESHOLD):
                return int(df["cum_timesteps"].iloc[i]), roll
        return None, roll
    for i in range(len(vals)):
        rest = vals[i:]
        if not np.isnan(rest).any() and np.min(rest) >= COMPETENCE_THRESHOLD:
            return int(df["cum_timesteps"].iloc[i]), roll
    return None, roll


def fig5_convergence():
    best_of = {}
    for algo in ["dqn", "reinforce", "a2c", "ppo"]:
        best_of[algo] = rank_by_final_reward(algo, available_combos(algo))[0]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ax = axes[0]
    crossings, crossings_window5 = {}, {}
    for algo in ["dqn", "reinforce", "a2c", "ppo"]:
        c = best_of[algo]
        df = load_monitor(algo, c)
        cross, roll = first_sustained_crossing(df, mode="holds")
        crossings[algo] = cross
        crossings_window5[algo] = first_sustained_crossing(df, mode="window5")[0]
        ax.plot(df["cum_timesteps"], roll, color=ALGO_COLOUR[algo], linewidth=1.4,
                label=f"{ALGO_LABEL[algo]} #{c}")
    ax.axhline(COMPETENCE_THRESHOLD, color="0.25", linestyle=(0, (5, 3)), linewidth=0.9)
    ax.annotate("competence threshold", xy=(0.98, COMPETENCE_THRESHOLD),
                xycoords=("axes fraction", "data"), ha="right", va="bottom",
                fontsize=6, color="0.25")
    ax.set_xlabel("environment timesteps")
    ax.set_ylabel("episode return\n(rolling mean, window = 20)")
    ax.set_title("Best configuration per algorithm")
    ax.legend(loc="lower right", framealpha=0.9, handlelength=1.6, fontsize=6)

    ax = axes[1]
    algos = ["dqn", "reinforce", "a2c", "ppo"]
    budgets = {a: int(load_monitor(a, best_of[a])["cum_timesteps"].iloc[-1]) for a in algos}
    cap = max(v for v in list(crossings.values()) + list(budgets.values()) if v is not None)
    xs, heights, hatches, colours = [], [], [], []
    for a in algos:
        xs.append(ALGO_LABEL[a])
        heights.append(cap if crossings[a] is None else crossings[a])
        hatches.append("//" if crossings[a] is None else "")
        colours.append(ALGO_COLOUR[a])
    bars = ax.bar(xs, heights, color=colours, edgecolor="white", linewidth=0.6)
    for bar, h, a in zip(bars, hatches, algos):
        bar.set_hatch(h)
        if crossings[a] is None:
            bar.set_alpha(0.35)
            ax.text(bar.get_x() + bar.get_width() / 2, cap * 0.5, "not reached\nin budget",
                    ha="center", va="center", fontsize=6, rotation=90, color="0.15")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, crossings[a], f"{crossings[a]:,}",
                    ha="center", va="bottom", fontsize=6)
    ax.set_ylabel("timesteps to sustained\nreturn >= 0 (held to end)")
    ax.set_title("Time to competence")
    ax.tick_params(axis="x", labelsize=6.5)
    fig.suptitle("Convergence comparison across algorithms", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT_DIR / "convergence.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out, best_of, crossings, budgets, crossings_window5


# ----------------------------------------------------------------------
# Figure 6 -- generalization
# ----------------------------------------------------------------------
def fig6_generalization():
    path = TABLES_DIR / "generalization_eval.csv"
    if not path.exists():
        return None, None
    t = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    x = np.arange(len(t))
    pct = t["success_rate"] * 100
    lo = pct - t["wilson_lo"] * 100
    hi = t["wilson_hi"] * 100 - pct
    ax.bar(x, pct, color=ALGO_COLOUR["ppo"], edgecolor="white", linewidth=0.6)
    ax.errorbar(x, pct, yerr=[lo, hi], fmt="none", ecolor="0.2", elinewidth=0.9, capsize=3)
    for xi, v, h in zip(x, pct, t["wilson_hi"] * 100):
        ax.text(xi, h + 2.0, f"{v:.0f}%", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c).replace(" (", "\n(") for c in t["condition"]], fontsize=6)
    ax.set_ylim(0, 105)
    ax.set_ylabel("plane-acquisition success rate (%)")
    n = int(t["n"].iloc[0])
    ax.set_title(f"Generalization by start condition (N = {n} per condition)\n"
                 f"error bars: Wilson 95% CI", fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / "generalization.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out, t


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Figure 2 ===")
    p, notes = fig2_reward_curves()
    print(f"  {p}")
    for line in notes:
        print(f"    {line}")

    print("=== Figure 3 ===")
    p, plotted = fig3_dqn_loss()
    print(f"  {p}  (configs plotted: {plotted})")

    print("=== Figure 4 ===")
    p, overlaps = fig4_pg_entropy()
    print(f"  {p}")
    print(f"    coincident REINFORCE pairs overlaid: {overlaps}")

    print("=== Figure 5 ===")
    p, best_of, crossings, budgets, crossings_w5 = fig5_convergence()
    print(f"  {p}")
    print(f"    best config per algo: {best_of}")
    print(f"    timesteps to sustained return>=0 (holds to end of run): {crossings}")
    print(f"    timesteps to return>=0 held 5 consecutive episodes: {crossings_w5}")
    print(f"    run budget (final cum_timesteps): {budgets}")

    print("=== Figure 6 ===")
    p, t = fig6_generalization()
    if p is None:
        print("  SKIPPED: logs/tables/generalization_eval.csv not found "
              "(run scripts/deployed_eval.py first)")
    else:
        print(f"  {p}")
        print(t.to_string(index=False))


if __name__ == "__main__":
    main()
