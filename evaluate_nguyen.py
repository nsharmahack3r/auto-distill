"""
evaluate_nguyen.py — Validation and plotting for Nguyen et al. comparison runs.

Mirrors evaluate_results.py exactly in structure and output format so that
results from both experiments live in the same CSV and can be plotted together.

Outputs
───────
nguyen_results.csv          — mAP per round for method1 and method2
nguyen_results_class.csv    — per-class mAP per round
nguyen_comparison.csv       — side-by-side with your Pipeline 2 results
                              (auto-merged from experiment_results.csv if present)
nguyen_graph_<ts>.png       — mAP@50 and mAP@50-95 curves for both methods
nguyen_comparison_<ts>.png  — comparison chart: Pipeline 2 vs Method 1 vs Method 2
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from ultralytics import YOLO
from notify import notify, notify_fail

EXPERIMENT_ROOT  = "nguyen_workspace"
VALIDATION_YAML  = "validation_config.yaml"
CLASS_NAMES      = ["rhino", "zebra", "leopard"]


METHOD_LABELS = {
    "method1": "Nguyen Method 1 (Div-Unc)",
    "method2": "Nguyen Method 2 (Unc-Div)",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _sort_key(path: str) -> int:
    try:
        return int(path.split("student_round_")[1].split("\\")[0].split("/")[0])
    except (IndexError, ValueError):
        return -1


def _find_checkpoints(method_name: str) -> list:
    pattern = f"{EXPERIMENT_ROOT}/{method_name}/models/student_round_*/weights/best.pt"
    paths   = glob.glob(pattern)
    paths.sort(key=_sort_key)
    return paths


def _evaluate_checkpoint(model_path: str, round_idx: int, label: str) -> dict | None:
    try:
        model   = YOLO(model_path)
        metrics = model.val(data=VALIDATION_YAML, verbose=False, workers=0)

        map50    = metrics.box.map50
        map50_95 = metrics.box.map

        # ── Per-class mAP extraction with single-class guard ─────────────
        # When a model is trained on fewer classes than the validation YAML
        # defines, metrics.box.maps contains only as many entries as the
        # model has output heads — not len(CLASS_NAMES).  Without this
        # guard, the loop silently reuses maps[0] for all missing classes,
        # producing identical values across all per-class columns (the bug
        # seen when running on a single-class dataset).
        #
        # Fix: only read maps[i] when i is a valid index.  Write None for
        # any class the model was not trained on.  Emit a console warning
        # so the mismatch is visible rather than hidden in the CSV.
        per_class = {f"mAP50_{name}": None for name in CLASS_NAMES}

        if hasattr(metrics.box, "maps") and metrics.box.maps is not None:
            n_model_classes = len(metrics.box.maps)
            n_yaml_classes  = len(CLASS_NAMES)

            if n_model_classes != n_yaml_classes:
                print(
                    f"  [WARNING] Round {round_idx}: model has {n_model_classes} "
                    f"output class(es) but validation YAML defines {n_yaml_classes}. "
                    f"Per-class mAP will be None for missing classes. "
                    f"Aggregate mAP@50 / mAP@50-95 are still valid."
                )

            for i, name in enumerate(CLASS_NAMES):
                if i < n_model_classes:
                    per_class[f"mAP50_{name}"] = float(metrics.box.maps[i])
                # else: already None — class not present in this model

        del model
        return {
            "Method"    : label,
            "Round"     : round_idx,
            "Model_Path": model_path,
            "mAP_50"    : map50,
            "mAP_50_95" : map50_95,
            **per_class,
        }

    except Exception as e:
        print(f"  Failed round {round_idx}: {e}")
        notify_fail(label, f"eval round {round_idx}", e)
        return None


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_nguyen(df: pd.DataFrame, ts: str) -> None:
    """mAP@50 and mAP@50-95 curves for Method 1 and Method 2."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"method1": "#2196F3", "method2": "#FF9800"}

    for method_key, label in METHOD_LABELS.items():
        sub = df[df["Method"] == label].sort_values("Round")
        if sub.empty:
            continue
        color = colors.get(method_key, "gray")
        ax.plot(sub["Round"], sub["mAP_50"],
                marker="o", linestyle="-", color=color,
                label=f"{label} — mAP@50")
        ax.plot(sub["Round"], sub["mAP_50_95"],
                marker="s", linestyle="--", color=color, alpha=0.6,
                label=f"{label} — mAP@50-95")

    ax.set_title("Nguyen et al. Methods — Performance over Active Rounds")
    ax.set_xlabel("Active Learning Round")
    ax.set_ylabel("mAP")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(sorted(df["Round"].unique()))

    fname = f"nguyen_graph_{ts}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Nguyen graph saved → {fname}")


def _plot_comparison(nguyen_df: pd.DataFrame, ts: str) -> None:
    """
    Overlay Nguyen Methods 1 & 2 with your Pipeline 2 (DFL) from
    experiment_results.csv, if that file exists.
    """
    pipeline2_csv = "experiment_results.csv"
    if not os.path.exists(pipeline2_csv):
        print("experiment_results.csv not found — skipping comparison plot.")
        return

    p2 = pd.read_csv(pipeline2_csv)
    # Rename 'Model' column to 'Method' for uniform plotting
    if "Model" in p2.columns:
        p2 = p2.rename(columns={"Model": "Method"})
    # Use only the first (or yolov8n) run from your results
    if "Method" in p2.columns:
        p2_label = p2["Method"].iloc[0]
    else:
        p2_label = "Pipeline 2 (DFL)"
    p2["Method"] = "Your Pipeline 2 (DFL)"

    combined = pd.concat([nguyen_df, p2], ignore_index=True)

    # Flag rows where per-class data is unreliable (single-class training run).
    # A row is flagged when all three per-class mAP columns are identical and
    # non-null — the signature of the silent copy-across bug — OR all are null.
    def _perclass_suspect(row):
        vals = [row.get(f"mAP50_{c}") for c in ["rhino", "zebra", "leopard"]]
        non_null = [v for v in vals if v is not None and not pd.isna(v)]
        if len(non_null) == 0:
            return "missing"
        if len(set(round(v, 8) for v in non_null)) == 1 and len(non_null) == 3:
            return "suspect_single_class"
        return "ok"

    combined["per_class_status"] = combined.apply(_perclass_suspect, axis=1)
    n_suspect = (combined["per_class_status"] != "ok").sum()
    if n_suspect > 0:
        print(
            f"  [WARNING] {n_suspect} row(s) in comparison CSV flagged as "
            f"'suspect_single_class' or 'missing' in per_class_status column. "
            f"These rows should not be used for per-class comparisons in the paper."
        )

    combined.to_csv("nguyen_comparison.csv", index=False)
    print("Comparison CSV saved → nguyen_comparison.csv")

    fig, ax = plt.subplots(figsize=(12, 7))

    style_map = {
        "Your Pipeline 2 (DFL)"          : ("#4CAF50", "-",  "o"),
        METHOD_LABELS["method1"]          : ("#2196F3", "--", "s"),
        METHOD_LABELS["method2"]          : ("#FF9800", ":",  "^"),
    }

    for method, (color, ls, marker) in style_map.items():
        sub = combined[combined["Method"] == method].sort_values("Round")
        if sub.empty:
            continue
        ax.plot(sub["Round"], sub["mAP_50"],
                marker=marker, linestyle=ls, color=color, linewidth=2,
                label=f"{method}")

    ax.set_title("Comparison: Your Pipeline 2 (DFL) vs Nguyen et al. Methods")
    ax.set_xlabel("Active Learning Round")
    ax.set_ylabel("mAP@50")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    rounds = sorted(combined["Round"].dropna().unique())
    ax.set_xticks(rounds)

    fname = f"nguyen_comparison_{ts}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison graph saved → {fname}")


# ── Public entry point ────────────────────────────────────────────────────────

def evaluate(method_name: str = None) -> pd.DataFrame:
    """
    Evaluate all Nguyen checkpoints.

    Parameters
    ----------
    method_name : str, optional
        "method1" or "method2". Evaluates both if None.

    Returns
    -------
    pd.DataFrame with all results.
    """
    methods = [method_name] if method_name else list(METHOD_LABELS.keys())
    notify("Nguyen Evaluation", f"Starting — methods={methods}")

    rows = []

    for mkey in methods:
        label      = METHOD_LABELS[mkey]
        checkpoints = _find_checkpoints(mkey)

        if not checkpoints:
            print(f"No checkpoints found for {mkey}. Did the experiment run?")
            continue

        print(f"\n--- Evaluating {label} ({len(checkpoints)} rounds) ---")

        for ckpt in checkpoints:
            ridx = _sort_key(ckpt)
            print(f"  Round {ridx}...")
            result = _evaluate_checkpoint(ckpt, ridx, label)
            if result:
                rows.append(result)
                print(f"    mAP@50={result['mAP_50']:.4f}  "
                      f"mAP@50-95={result['mAP_50_95']:.4f}")

    if not rows:
        print("No results collected.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Console summary — only show per-class columns that have at least one
    # non-None value (avoids cluttered all-None columns on single-class runs)
    print("\n=== NGUYEN RESULTS ===")
    per_class_cols = [
        f"mAP50_{c}" for c in CLASS_NAMES
        if f"mAP50_{c}" in df.columns and df[f"mAP50_{c}"].notna().any()
    ]
    show_cols = ["Method", "Round", "mAP_50", "mAP_50_95"] + per_class_cols
    print(df[show_cols].to_string(index=False))

    # Warn if any per-class columns are entirely None
    null_cols = [
        f"mAP50_{c}" for c in CLASS_NAMES
        if f"mAP50_{c}" in df.columns and df[f"mAP50_{c}"].isna().all()
    ]
    if null_cols:
        print(
            f"\n  [WARNING] The following per-class columns are all None — "
            f"the model was likely trained on fewer classes than the validation YAML: "
            f"{null_cols}"
        )

    # Save
    df.to_csv("nguyen_results.csv", index=False)
    class_cols = [c for c in df.columns if "mAP50_" in c or c in ("Method", "Round")]
    df[class_cols].to_csv("nguyen_results_class.csv", index=False)
    print("\nSaved → nguyen_results.csv")
    print("Saved → nguyen_results_class.csv")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _plot_nguyen(df, ts)
    _plot_comparison(df, ts)

    notify("Nguyen Evaluation complete", f"rounds={len(rows)}")
    return df


if __name__ == "__main__":
    evaluate()