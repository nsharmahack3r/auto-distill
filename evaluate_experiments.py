"""
evaluate_experiments.py — Publication-quality plots for IEEE paper experiments.

Three plot functions, one per experiment:
    plot_ablation(df, out_dir)             — mAP per round for loop ablation
    plot_strategy_comparison(df, out_dir)  — mAP per round for each query strategy
    plot_budget_sensitivity(df, out_dir)   — mAP per round for each budget level

All figures use a consistent style suitable for IEEE two-column format.
"""

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from datetime import datetime

# ── Plot style ────────────────────────────────────────────────────────────────

# IEEE-friendly defaults: serif font, tight layout, high DPI
STYLE = {
    "font.family":     "serif",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi":      200,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "axes.grid":       True,
    "grid.alpha":      0.35,
    "grid.linestyle":  "--",
}

COLORS = {
    "dfl_variance":      "#2563EB",   # blue
    "random":            "#DC2626",   # red
    "least_confidence":  "#16A34A",   # green
}

BUDGET_COLORS = {
    400:  "#F59E0B",   # amber
    800:  "#8B5CF6",   # violet
    1200: "#0891B2",   # cyan
}

MARKERS = ["o", "s", "^", "D", "v", "P"]


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 1 — Loop Ablation
# ══════════════════════════════════════════════════════════════════════════════

def plot_ablation(df: pd.DataFrame, out_dir: str) -> str:
    """
    Line plot: mAP@50 and mAP@50-95 over active-learning rounds.

    Parameters
    ----------
    df : DataFrame with columns [Round, mAP_50, mAP_50_95]
    out_dir : directory to save the figure

    Returns
    -------
    Path to saved figure.
    """
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 4))

        df = df.sort_values("Round")

        ax.plot(df["Round"], df["mAP_50"],
                marker="o", linewidth=2, color="#2563EB",
                label="mAP@50", zorder=3)

        ax.plot(df["Round"], df["mAP_50_95"],
                marker="s", linewidth=2, linestyle="--", color="#7C3AED",
                label="mAP@50-95", alpha=0.85, zorder=3)

        # Annotate peak mAP@50
        best_idx = df["mAP_50"].idxmax()
        best_row = df.loc[best_idx]
        ax.annotate(
            f'{best_row["mAP_50"]:.3f}',
            xy=(best_row["Round"], best_row["mAP_50"]),
            xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=9, fontweight="bold", color="#2563EB",
            arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.2),
        )

        ax.set_xlabel("Active Learning Round")
        ax.set_ylabel("Mean Average Precision")
        ax.set_title("Loop Ablation — Performance per Round")
        ax.set_xticks(sorted(df["Round"].unique()))
        ax.legend(loc="lower right", framealpha=0.9)
        ax.set_ylim(bottom=0)

        fname = os.path.join(out_dir, f"ablation_curve_{_ts()}.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Ablation plot saved → {fname}")
        return fname


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 2 — Query Strategy Comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_strategy_comparison(df: pd.DataFrame, out_dir: str) -> str:
    """
    Multi-line plot: mAP@50 over rounds for each query strategy.

    Parameters
    ----------
    df : DataFrame with columns [Round, mAP_50, Strategy]
    out_dir : directory to save the figure

    Returns
    -------
    Path to saved figure.
    """
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4.5))

        strategy_labels = {
            "dfl_variance":     "DFL Variance (Ours)",
            "random":           "Random Sampling",
            "least_confidence": "Least Confidence",
        }

        for i, (strategy, group) in enumerate(df.groupby("Strategy")):
            group = group.sort_values("Round")
            label = strategy_labels.get(strategy, strategy)
            color = COLORS.get(strategy, f"C{i}")
            ax.plot(
                group["Round"], group["mAP_50"],
                marker=MARKERS[i % len(MARKERS)],
                linewidth=2, markersize=7,
                color=color, label=label, zorder=3,
            )

        ax.set_xlabel("Active Learning Round")
        ax.set_ylabel("mAP@50")
        ax.set_title("Query Strategy Comparison — mAP@50 per Round")
        ax.set_xticks(sorted(df["Round"].unique()))
        ax.legend(loc="lower right", framealpha=0.9)
        ax.set_ylim(bottom=0)

        fname = os.path.join(out_dir, f"strategy_comparison_{_ts()}.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Strategy comparison plot saved → {fname}")
        return fname


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 3 — Budget Sensitivity
# ══════════════════════════════════════════════════════════════════════════════

def plot_budget_sensitivity(df: pd.DataFrame, out_dir: str) -> str:
    """
    Multi-line plot: mAP@50 over rounds for each budget level,
    plus a grouped bar chart of final-round mAP.

    Parameters
    ----------
    df : DataFrame with columns [Round, mAP_50, mAP_50_95, Budget]
    out_dir : directory to save the figure

    Returns
    -------
    Path to saved figure (multi-line).
    """
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5),
                                 gridspec_kw={"width_ratios": [2, 1]})

        # ── Left panel: line plot over rounds ─────────────────────────────
        ax = axes[0]
        for i, (budget, group) in enumerate(df.groupby("Budget")):
            group = group.sort_values("Round")
            color = BUDGET_COLORS.get(budget, f"C{i}")
            ax.plot(
                group["Round"], group["mAP_50"],
                marker=MARKERS[i % len(MARKERS)],
                linewidth=2, markersize=7,
                color=color, label=f"{budget} images", zorder=3,
            )

        ax.set_xlabel("Active Learning Round")
        ax.set_ylabel("mAP@50")
        ax.set_title("Budget Sensitivity — mAP@50 per Round")
        ax.set_xticks(sorted(df["Round"].unique()))
        ax.legend(loc="lower right", framealpha=0.9)
        ax.set_ylim(bottom=0)

        # ── Right panel: bar chart of final-round mAP ────────────────────
        ax2 = axes[1]
        final_rows = df.loc[df.groupby("Budget")["Round"].idxmax()]
        final_rows = final_rows.sort_values("Budget")

        bars = ax2.bar(
            [str(b) for b in final_rows["Budget"]],
            final_rows["mAP_50"],
            color=[BUDGET_COLORS.get(b, "#6B7280") for b in final_rows["Budget"]],
            edgecolor="white", linewidth=1.5, width=0.6,
        )

        # Value labels on bars
        for bar, val in zip(bars, final_rows["mAP_50"]):
            ax2.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

        ax2.set_xlabel("Annotation Budget")
        ax2.set_ylabel("Final mAP@50")
        ax2.set_title("Final Performance")
        ax2.set_ylim(0, max(final_rows["mAP_50"]) * 1.15)

        fig.suptitle("Budget Sensitivity Analysis", fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()

        fname = os.path.join(out_dir, f"budget_sensitivity_{_ts()}.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Budget sensitivity plot saved → {fname}")
        return fname


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Re-plot experiment results from CSVs.")
    parser.add_argument("experiment", choices=["ablation", "strategy", "budget"],
                        help="Which experiment to re-plot.")
    parser.add_argument("--csv", required=True, help="Path to the results CSV.")
    parser.add_argument("--out", default="results", help="Output directory.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(args.out, exist_ok=True)

    if args.experiment == "ablation":
        plot_ablation(df, args.out)
    elif args.experiment == "strategy":
        plot_strategy_comparison(df, args.out)
    elif args.experiment == "budget":
        plot_budget_sensitivity(df, args.out)
