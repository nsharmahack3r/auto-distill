"""
evaluate_results.py — Validation and result plotting for all model runs.

Produces:
  experiment_results.csv        — aggregate mAP per round, per model
  experiment_results_class.csv  — per-class mAP per round, per model
  accuracy_graph_<ts>.png       — mAP@50 and mAP@50-95 curves
  perclass_graph_<ts>.png       — per-class mAP@50 curves
"""

import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

from notify import notify, notify_fail

EXPERIMENT_ROOT  = "experiment_workspace"
VALIDATION_YAML  = "validation_config.yaml"
CLASS_NAMES      = ["rhino", "zebra", "leopard"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sort_key(path: str) -> int:
    """Extract integer round index from a model path."""
    try:
        return int(path.split("student_round_")[1].split("\\")[0].split("/")[0])
    except (IndexError, ValueError):
        return -1


def _find_models(model_name: str = None) -> list:
    """
    Return sorted best.pt paths under EXPERIMENT_ROOT.
    If model_name is given, scope to that model's subfolder.
    """
    pattern = (
        f"{EXPERIMENT_ROOT}/{model_name}/models/student_round_*/weights/best.pt"
        if model_name
        else f"{EXPERIMENT_ROOT}/models/student_round_*/weights/best.pt"
    )
    paths = glob.glob(pattern)
    paths.sort(key=_sort_key)
    return paths


# ── Core evaluation ───────────────────────────────────────────────────────────

def _evaluate_model(model_path: str, round_name: str, model_label: str) -> dict | None:
    """
    Run validation on one checkpoint.
    Returns a result dict, or None on failure.
    """
    try:
        model   = YOLO(model_path)
        metrics = model.val(data=VALIDATION_YAML, verbose=False, workers=0)

        # Aggregate metrics
        map50     = metrics.box.map50
        map50_95  = metrics.box.map

        # Per-class metrics (maps is an array aligned to class indices)
        per_class = {}
        if hasattr(metrics.box, "maps") and metrics.box.maps is not None:
            for i, name in enumerate(CLASS_NAMES):
                per_class[f"mAP50_{name}"] = (
                    float(metrics.box.maps[i]) if i < len(metrics.box.maps) else None
                )

        del model

        return {
            "Model"      : model_label,
            "Round"      : int(round_name),
            "Model_Path" : model_path,
            "mAP_50"     : map50,
            "mAP_50_95"  : map50_95,
            **per_class,
        }

    except Exception as e:
        print(f"  Failed to evaluate round {round_name}: {e}")
        notify_fail(model_label, f"evaluation round {round_name}", e)
        return None


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_aggregate(df: pd.DataFrame, timestamp: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_label, group in df.groupby("Model"):
        group = group.sort_values("Round")
        ax.plot(group["Round"], group["mAP_50"],
                marker="o", linestyle="-",  label=f"{model_label} mAP@50")
        ax.plot(group["Round"], group["mAP_50_95"],
                marker="s", linestyle="--", label=f"{model_label} mAP@50-95", alpha=0.7)

    ax.set_title("Active Distillation — Performance over Rounds")
    ax.set_xlabel("Active Learning Round")
    ax.set_ylabel("Mean Average Precision (mAP)")
    ax.grid(True, alpha=0.4)
    ax.legend()
    ax.set_xticks(sorted(df["Round"].unique()))

    fname = f"accuracy_graph_{timestamp}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Aggregate graph saved → {fname}")


def _plot_per_class(df: pd.DataFrame, timestamp: str) -> None:
    class_cols = [f"mAP50_{c}" for c in CLASS_NAMES if f"mAP50_{c}" in df.columns]
    if not class_cols:
        print("Per-class columns not found; skipping per-class plot.")
        return

    fig, axes = plt.subplots(1, len(class_cols), figsize=(5 * len(class_cols), 5), sharey=True)
    if len(class_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, class_cols):
        class_name = col.replace("mAP50_", "")
        for model_label, group in df.groupby("Model"):
            group = group.sort_values("Round")
            ax.plot(group["Round"], group[col], marker="o", label=model_label)
        ax.set_title(class_name.capitalize())
        ax.set_xlabel("Round")
        ax.set_ylabel("mAP@50")
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8)

    fig.suptitle("Per-Class mAP@50 over Active Learning Rounds")
    fname = f"perclass_graph_{timestamp}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Per-class graph saved  → {fname}")


# ── Public entry point ────────────────────────────────────────────────────────

def evaluate(model_name: str = None) -> pd.DataFrame:
    """
    Evaluate all checkpoints under EXPERIMENT_ROOT.

    Parameters
    ----------
    model_name : str, optional
        Scope evaluation to a specific model subfolder
        (e.g. "rtdetr_l"). Evaluates all models if None.

    Returns
    -------
    pd.DataFrame with all results.
    """
    notify("Evaluation", f"Starting evaluation — model={model_name or 'all'}")
    print("\n--- Starting Evaluation ---")

    model_paths = _find_models(model_name)

    if not model_paths:
        print("No model checkpoints found. Did the experiment run successfully?")
        return pd.DataFrame()

    label = model_name or "experiment"
    rows  = []

    for model_path in model_paths:
        round_name = str(_sort_key(model_path))
        print(f"  Evaluating round {round_name} [{label}]...")
        result = _evaluate_model(model_path, round_name, label)
        if result:
            rows.append(result)
            print(f"    mAP@50={result['mAP_50']:.4f}  mAP@50-95={result['mAP_50_95']:.4f}")

    if not rows:
        print("No results collected.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Console summary
    print("\n=== RESULTS ===")
    display_cols = ["Model", "Round", "mAP_50", "mAP_50_95"] + \
                   [f"mAP50_{c}" for c in CLASS_NAMES if f"mAP50_{c}" in df.columns]
    print(df[display_cols].to_string(index=False))

    # Save CSVs
    df.to_csv("experiment_results.csv", index=False)
    df[[c for c in df.columns if "mAP50_" in c or c in ("Model", "Round")]].to_csv(
        "experiment_results_class.csv", index=False
    )
    print("\nResults saved → experiment_results.csv")
    print("Per-class   → experiment_results_class.csv")

    # Plots
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _plot_aggregate(df, ts)
    _plot_per_class(df, ts)

    notify("Evaluation complete", f"model={label}  rounds={len(rows)}")
    return df


if __name__ == "__main__":
    evaluate()
