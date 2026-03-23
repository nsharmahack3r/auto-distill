"""
run_experiments.py — Experiment runner for IEEE paper evaluation.

Three experiments, each reusing the existing active-distillation pipeline:

  1. ablation   — Loop ablation: mAP per round with default settings
  2. strategy   — Query strategy comparison: DFL-variance vs random vs least-confidence
  3. budget     — Budget sensitivity: 400 / 800 / 1200 image pools

Usage
-----
    python run_experiments.py ablation  --model yolov8n
    python run_experiments.py strategy  --model yolov8n
    python run_experiments.py budget    --model yolov8n
    python run_experiments.py all       --model yolov8n
    python run_experiments.py ablation  --dry-run          # preview config only
"""

import os
import gc
import glob
import shutil
import random
import argparse
import torch
import pandas as pd

from ultralytics import YOLO
from tqdm import tqdm

from novelity_sampler import DFLUncertaintySampler
from active_sampler   import UncertaintySampler
from teacher_labeler  import TeacherLabeler
from dataset          import DatasetInfo
from notify import (
    Timer,
    notify_start,
    notify_phase,
    notify_loop,
    notify_done,
    notify_fail,
)
import evaluate_experiments

# ── Paths (same as main_experiment.py) ────────────────────────────────────────

RAW_IMAGE_DIR   = DatasetInfo.imagesPath
EXPERIMENT_ROOT = "experiment_workspace"
GD_CONFIG       = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_WEIGHTS      = "groundingdino_swint_ogc.pth"
RESULTS_ROOT    = "results"

CLASS_MAP = {
    "rhino"  : 0,
    "zebra"  : 1,
    "leopard": 2,
}

# ── Active-learning hyper-parameters ─────────────────────────────────────────

INITIAL_SEED_PCT    = 0.10
ACTIVE_BATCH_PCT    = 0.05
MAX_LOOPS           = 6
COLD_START_EPOCHS   = 10
ACTIVE_TRAIN_EPOCHS = 20

# ── Per-model batch sizes ────────────────────────────────────────────────────

DEFAULT_BATCH = 16
MODEL_BATCH = {
    "rtdetr-l": 8,
    "rtdetr-x": 4,
    "yolov8l" : 8,
    "yolov8x" : 4,
}

MODEL_REGISTRY = {
    "yolov8n" : "yolov8n.pt",
    "yolov8s" : "yolov8s.pt",
    "yolov9s" : "yolov9s.pt",
    "yolov10n": "yolov10n.pt",
    "yolo11n" : "yolo11n.pt",
    "rtdetr-l": "rtdetr-l.pt",
    "rtdetr-x": "rtdetr-x.pt",
}

# ── Budget levels for Experiment 3 ───────────────────────────────────────────

BUDGET_LEVELS = [400, 800, 1200]

# ── Strategy definitions for Experiment 2 ────────────────────────────────────

STRATEGY_NAMES = ["dfl_variance", "random", "least_confidence"]

VALIDATION_YAML = "validation_config.yaml"


# ══════════════════════════════════════════════════════════════════════════════
#  Random sampler (baseline for strategy comparison)
# ══════════════════════════════════════════════════════════════════════════════

class RandomSampler:
    """Selects images uniformly at random — no model needed."""

    def __init__(self, model_path: str = None):
        # Accept model_path for API compatibility; it is ignored.
        pass

    def select_batch(self, image_list: list, batch_size: int) -> list:
        print(f"Selecting {batch_size} images at random from {len(image_list)}...")
        return random.sample(image_list, min(batch_size, len(image_list)))

    def cleanup(self):
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  Shared helpers (mirrors main_experiment.py but parameterised)
# ══════════════════════════════════════════════════════════════════════════════

def force_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def setup_workspace(run_tag: str, image_budget: int | None = None) -> list:
    """
    Create a workspace under experiment_workspace/<run_tag>/,
    copy images from the raw dataset into pool_images/.

    Parameters
    ----------
    run_tag : str
        Unique workspace name (e.g. "ablation_yolov8n", "strategy_random_yolov8n").
    image_budget : int or None
        If set, only copy this many images (randomly sampled) into the pool.
    """
    model_root = os.path.join(EXPERIMENT_ROOT, run_tag)

    if os.path.exists(model_root):
        try:
            shutil.rmtree(model_root)
        except Exception:
            print(f"Warning: could not remove {model_root}")

    os.makedirs(f"{model_root}/pool_images",       exist_ok=True)
    os.makedirs(f"{model_root}/train_data/images",  exist_ok=True)
    os.makedirs(f"{model_root}/train_data/labels",  exist_ok=True)

    all_images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        all_images.extend(glob.glob(os.path.join(RAW_IMAGE_DIR, ext)))

    if image_budget is not None and image_budget < len(all_images):
        all_images = random.sample(all_images, image_budget)
        print(f"Budget cap: using {image_budget} / {len(all_images)} images")

    for img in all_images:
        shutil.copy(img, f"{model_root}/pool_images/")

    pool = glob.glob(f"{model_root}/pool_images/*.*")
    print(f"Workspace ready: {model_root}  ({len(pool)} images)")
    return pool


def create_yaml(output_folder: str) -> str:
    abs_path = os.path.abspath(output_folder).replace("\\", "/")
    yaml_content = (
        f"path: {abs_path}\n"
        f"train: images\n"
        f"val: images\n"
        f"names:\n"
        f"  0: rhino\n"
        f"  1: zebra\n"
        f"  2: leopard\n"
    )
    yaml_path = os.path.join(output_folder, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    return os.path.abspath(yaml_path)


def train_model(model_id, run_tag, yaml_path, epochs, round_idx):
    batch      = MODEL_BATCH.get(run_tag.split("_")[-1], DEFAULT_BATCH)
    model_root = os.path.join(EXPERIMENT_ROOT, run_tag)
    run_name   = f"student_round_{round_idx}"
    project    = os.path.abspath(f"{model_root}/models")

    model = YOLO(model_id)
    model.train(
        data    = yaml_path,
        epochs  = epochs,
        imgsz   = 640,
        batch   = batch,
        workers = 2 if round_idx == 0 else 1,
        project = project,
        name    = run_name,
        verbose = False,
    )
    best_pt = os.path.join(project, run_name, "weights", "best.pt")

    del model
    force_cleanup()
    return best_pt


def quick_val(model_path: str) -> tuple[float, float]:
    """Return (mAP50, mAP50-95) for a checkpoint."""
    model   = YOLO(model_path)
    metrics = model.val(data=VALIDATION_YAML, verbose=False, workers=0)
    result  = (metrics.box.map50, metrics.box.map)
    del model
    force_cleanup()
    return result


def evaluate_run(run_tag: str) -> pd.DataFrame:
    """Evaluate all checkpoints in a run and return a DataFrame."""
    pattern = os.path.join(EXPERIMENT_ROOT, run_tag, "models", "student_round_*", "weights", "best.pt")
    paths   = sorted(glob.glob(pattern), key=_sort_key)

    rows = []
    for p in paths:
        rnd = _sort_key(p)
        try:
            map50, map50_95 = quick_val(p)
            rows.append({
                "Run":       run_tag,
                "Round":     rnd,
                "mAP_50":    map50,
                "mAP_50_95": map50_95,
                "Model_Path": p,
            })
            print(f"  Round {rnd}: mAP@50={map50:.4f}  mAP@50-95={map50_95:.4f}")
        except Exception as e:
            print(f"  Round {rnd}: FAILED — {e}")
    return pd.DataFrame(rows)


def _sort_key(path: str) -> int:
    try:
        return int(path.split("student_round_")[1].split("\\")[0].split("/")[0])
    except (IndexError, ValueError):
        return -1


def _make_sampler(strategy: str, model_path: str):
    """Factory for sampler selection."""
    if strategy == "dfl_variance":
        return DFLUncertaintySampler(model_path)
    elif strategy == "random":
        return RandomSampler(model_path)
    elif strategy == "least_confidence":
        return UncertaintySampler(model_path)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _cleanup_sampler(sampler, strategy: str):
    """Strategy-aware cleanup."""
    if strategy == "dfl_variance":
        sampler.cleanup()
    elif strategy == "least_confidence":
        del sampler.model
    # RandomSampler has no resources to free


# ══════════════════════════════════════════════════════════════════════════════
#  Core pipeline (parameterised version of main_experiment.run_pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    run_tag      : str,
    model_name   : str,
    strategy     : str  = "dfl_variance",
    image_budget : int | None = None,
    max_loops    : int  = MAX_LOOPS,
) -> pd.DataFrame:
    """
    Execute the active-distillation pipeline for one configuration.

    Parameters
    ----------
    run_tag      : unique workspace name
    model_name   : key in MODEL_REGISTRY (e.g. "yolov8n")
    strategy     : "dfl_variance" | "random" | "least_confidence"
    image_budget : cap on pool size (None = use all images)
    max_loops    : number of active-learning loops

    Returns
    -------
    pd.DataFrame with per-round evaluation results.
    """
    model_id   = MODEL_REGISTRY[model_name]
    model_root = os.path.join(EXPERIMENT_ROOT, run_tag)

    force_cleanup()
    pool_images = setup_workspace(run_tag, image_budget=image_budget)
    yaml_path   = create_yaml(f"{model_root}/train_data")

    timer = notify_start(
        run_tag, f"Experiment ({strategy}, budget={image_budget or 'all'})",
        len(pool_images),
    )

    teacher = TeacherLabeler(
        model_config_path  = GD_CONFIG,
        model_weights_path = GD_WEIGHTS,
        class_map          = CLASS_MAP,
        output_folder      = f"{model_root}/train_data",
    )

    try:
        # ── Phase 1: Cold start ───────────────────────────────────────────
        num_seed   = max(1, int(len(pool_images) * INITIAL_SEED_PCT))
        seed_batch = random.sample(pool_images, num_seed)

        print(f"\n--- Cold Start ({num_seed} images) [{run_tag}] ---")
        notify_phase(run_tag, "Cold start — labelling", f"{num_seed} images", timer)

        teacher.label_batch(seed_batch)
        force_cleanup()

        for img in seed_batch:
            if os.path.exists(img):
                os.remove(img)

        # ── Round 0: train on seed ────────────────────────────────────────
        print(f"\n--- Training Round 0 [{run_tag}] ---")
        notify_phase(run_tag, "Training round 0", f"epochs={COLD_START_EPOCHS}", timer)

        current_model_path = train_model(
            model_id  = model_id,
            run_tag   = run_tag,
            yaml_path = yaml_path,
            epochs    = COLD_START_EPOCHS,
            round_idx = 0,
        )

        # ── Active loops ──────────────────────────────────────────────────
        for loop_idx in range(1, max_loops + 1):
            print(f"\n--- Active Loop {loop_idx}/{max_loops} [{run_tag}] ---")

            current_pool = glob.glob(f"{model_root}/pool_images/*.*")
            if not current_pool:
                print("Pool exhausted — stopping early.")
                break

            # Score pool with selected strategy
            notify_phase(run_tag, f"Loop {loop_idx} — scoring ({strategy})",
                         f"{len(current_pool)} images", timer)

            sampler    = _make_sampler(strategy, current_model_path)
            num_active = max(1, int(len(pool_images) * ACTIVE_BATCH_PCT))
            hard_batch = sampler.select_batch(current_pool, batch_size=num_active)
            _cleanup_sampler(sampler, strategy)
            del sampler
            force_cleanup()

            # Label selected batch
            notify_phase(run_tag, f"Loop {loop_idx} — labelling",
                         f"{len(hard_batch)} images", timer)
            teacher.label_batch(hard_batch)

            for img in hard_batch:
                if os.path.exists(img):
                    os.remove(img)

            # Retrain
            print(f"Retraining [{run_tag}] round {loop_idx}...")
            notify_phase(run_tag, f"Loop {loop_idx} — retraining",
                         f"epochs={ACTIVE_TRAIN_EPOCHS}", timer)

            current_model_path = train_model(
                model_id  = current_model_path,
                run_tag   = run_tag,
                yaml_path = yaml_path,
                epochs    = ACTIVE_TRAIN_EPOCHS,
                round_idx = loop_idx,
            )

            # Quick validation for ntfy updates
            try:
                map50, map50_95 = quick_val(current_model_path)
                notify_loop(run_tag, loop_idx, max_loops, map50, map50_95, timer)
            except Exception as val_err:
                print(f"Inline validation failed: {val_err}")

        # ── Final evaluation ──────────────────────────────────────────────
        notify_phase(run_tag, "Final evaluation", "", timer)
        df = evaluate_run(run_tag)

        if not df.empty:
            best_row   = df.loc[df["mAP_50"].idxmax()]
            best_map50 = float(best_row["mAP_50"])
            best_round = int(best_row["Round"])
            notify_done(run_tag, best_map50, best_round, timer)

        return df

    except Exception as e:
        notify_fail(run_tag, "pipeline", e, timer)
        raise


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 1 — Loop Ablation
# ══════════════════════════════════════════════════════════════════════════════

def run_ablation(model_name: str, dry_run: bool = False):
    """
    Run the standard pipeline and record mAP at every round.
    This is the default DFL-variance pipeline — the ablation is in the
    per-round reporting, not in varying any parameter.
    """
    run_tag   = f"ablation_{model_name}"
    out_dir   = os.path.join(RESULTS_ROOT, "ablation")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"  EXPERIMENT 1 — Loop Ablation  [{model_name}]")
    print("=" * 60)

    if dry_run:
        print(f"  Run tag       : {run_tag}")
        print(f"  Model         : {model_name} → {MODEL_REGISTRY[model_name]}")
        print(f"  Strategy      : dfl_variance")
        print(f"  Seed fraction : {INITIAL_SEED_PCT}")
        print(f"  Batch fraction: {ACTIVE_BATCH_PCT}")
        print(f"  Max loops     : {MAX_LOOPS}")
        print(f"  Output dir    : {out_dir}")
        print("  (dry-run — no training)")
        return

    df = run_pipeline(run_tag=run_tag, model_name=model_name, strategy="dfl_variance")

    csv_path = os.path.join(out_dir, "ablation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nAblation results saved → {csv_path}")

    evaluate_experiments.plot_ablation(df, out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 2 — Query Strategy Comparison
# ══════════════════════════════════════════════════════════════════════════════

def run_strategy_comparison(model_name: str, dry_run: bool = False):
    """
    Run the pipeline three times (DFL-variance, random, least-confidence),
    keeping everything else identical.
    """
    out_dir = os.path.join(RESULTS_ROOT, "strategy")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"  EXPERIMENT 2 — Query Strategy Comparison  [{model_name}]")
    print("=" * 60)

    if dry_run:
        for s in STRATEGY_NAMES:
            print(f"  Strategy: {s}  →  run tag: strategy_{s}_{model_name}")
        print(f"  Model       : {model_name} → {MODEL_REGISTRY[model_name]}")
        print(f"  Max loops   : {MAX_LOOPS}")
        print(f"  Output dir  : {out_dir}")
        print("  (dry-run — no training)")
        return

    all_dfs = []
    for strategy in STRATEGY_NAMES:
        run_tag = f"strategy_{strategy}_{model_name}"
        print(f"\n{'─' * 60}")
        print(f"  Running strategy: {strategy}")
        print(f"{'─' * 60}")

        df = run_pipeline(
            run_tag    = run_tag,
            model_name = model_name,
            strategy   = strategy,
        )
        df["Strategy"] = strategy
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    csv_path = os.path.join(out_dir, "strategy_results.csv")
    combined.to_csv(csv_path, index=False)
    print(f"\nStrategy results saved → {csv_path}")

    evaluate_experiments.plot_strategy_comparison(combined, out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 3 — Budget Sensitivity
# ══════════════════════════════════════════════════════════════════════════════

def run_budget_sensitivity(model_name: str, dry_run: bool = False):
    """
    Run the DFL-variance pipeline with different pool sizes
    (400, 800, 1200 images) to measure budget impact.
    """
    out_dir = os.path.join(RESULTS_ROOT, "budget")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"  EXPERIMENT 3 — Budget Sensitivity  [{model_name}]")
    print("=" * 60)

    if dry_run:
        for b in BUDGET_LEVELS:
            print(f"  Budget: {b}  →  run tag: budget_{b}_{model_name}")
        print(f"  Model       : {model_name} → {MODEL_REGISTRY[model_name]}")
        print(f"  Strategy    : dfl_variance")
        print(f"  Max loops   : {MAX_LOOPS}")
        print(f"  Output dir  : {out_dir}")
        print("  (dry-run — no training)")
        return

    all_dfs = []
    for budget in BUDGET_LEVELS:
        run_tag = f"budget_{budget}_{model_name}"
        print(f"\n{'─' * 60}")
        print(f"  Running budget: {budget} images")
        print(f"{'─' * 60}")

        df = run_pipeline(
            run_tag      = run_tag,
            model_name   = model_name,
            strategy     = "dfl_variance",
            image_budget = budget,
        )
        df["Budget"] = budget
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    csv_path = os.path.join(out_dir, "budget_results.csv")
    combined.to_csv(csv_path, index=False)
    print(f"\nBudget results saved → {csv_path}")

    evaluate_experiments.plot_budget_sensitivity(combined, out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ablation / strategy / budget experiments for the IEEE paper.",
    )
    parser.add_argument(
        "experiment",
        choices=["ablation", "strategy", "budget", "all"],
        help="Which experiment to run.",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default="yolov8n",
        help="Model to use (default: yolov8n).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed for reproducibility across all experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n  Random seed: {args.seed}")
    print(f"  Model      : {args.model}")
    print()

    funcs = {
        "ablation" : run_ablation,
        "strategy" : run_strategy_comparison,
        "budget"   : run_budget_sensitivity,
    }

    if args.experiment == "all":
        for name, func in funcs.items():
            func(args.model, dry_run=args.dry_run)
    else:
        funcs[args.experiment](args.model, dry_run=args.dry_run)

    print("\n✓ Experiment(s) complete.")


if __name__ == "__main__":
    main()
