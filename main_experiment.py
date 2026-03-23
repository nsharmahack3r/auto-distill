"""
main_experiment.py — Active distillation pipeline with multi-model support.

Supported model families
------------------------
  YOLOv8   : yolov8n / yolov8s / yolov8m / yolov8l / yolov8x
  YOLOv9   : yolov9s / yolov9m / yolov9c
  YOLOv10  : yolov10n / yolov10s / yolov10m
  YOLOv11  : yolo11n / yolo11s / yolo11m
  RT-DETR  : rtdetr-l / rtdetr-x

All paths from dataset.py and validation_config.yaml are kept intact
— this file is a drop-in replacement for the original main_experiment.py.

Usage
-----
    # Run one model:
    python main_experiment.py --model yolov8n

    # Run all models in MODEL_REGISTRY sequentially:
    python main_experiment.py --all

    # Run the original default (yolov8n) with no flags:
    python main_experiment.py
"""

import os
import gc
import glob
import shutil
import random
import argparse
import torch

from ultralytics import YOLO

from active_sampler   import UncertaintySampler
from novelity_sampler import DFLUncertaintySampler
from teacher_labeler  import TeacherLabeler
from dataset          import DatasetInfo
import evaluate_results
from notify import (
    Timer,
    notify_start,
    notify_phase,
    notify_loop,
    notify_done,
    notify_fail,
)

# ── Paths (unchanged from original) ──────────────────────────────────────────

RAW_IMAGE_DIR    = DatasetInfo.imagesPath
EXPERIMENT_ROOT  = "experiment_workspace"
GD_CONFIG        = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_WEIGHTS       = "groundingdino_swint_ogc.pth"

CLASS_MAP = {
    "rhino"  : 0,
    "zebra"  : 1,
    "leopard": 2,
}

# ── Active-learning hyper-parameters ─────────────────────────────────────────

INITIAL_SEED_PCT   = 0.10   # fraction of pool used for cold-start
ACTIVE_BATCH_PCT   = 0.05   # fraction added per active loop
MAX_LOOPS          = 6
COLD_START_EPOCHS  = 10
ACTIVE_TRAIN_EPOCHS = 20

# ── Model registry ────────────────────────────────────────────────────────────
#
# Each entry maps a short name to the Ultralytics model identifier.
# Add or remove entries here to control which models run with --all.
#
# Format:  "short_name": "ultralytics_model_id"
#
# Notes:
#   • YOLOv8/9/10/11 use the standard DFL head; DFLUncertaintySampler works natively.
#   • RT-DETR uses a transformer decoder; DFLUncertaintySampler falls back to
#     confidence-based scoring for the sampler (see novelity_sampler.py).
#   • Larger models (yolov8l, rtdetr-x) need more VRAM; reduce batch size if OOM.

MODEL_REGISTRY = {
    # ── YOLO family ──
    "yolov8n"  : "yolov8n.pt",
    "yolov8s"  : "yolov8s.pt",
    "yolov9s"  : "yolov9s.pt",
    "yolov10n" : "yolov10n.pt",
    "yolo11n"  : "yolo11n.pt",
    # ── Transformer family ──
    "rtdetr-l" : "rtdetr-l.pt",
    "rtdetr-x" : "rtdetr-x.pt",
}

# Per-model batch sizes — reduces VRAM pressure for larger models.
# Falls back to DEFAULT_BATCH for any model not listed.
DEFAULT_BATCH = 16
MODEL_BATCH = {
    "rtdetr-l" : 8,
    "rtdetr-x" : 4,
    "yolov8l"  : 8,
    "yolov8x"  : 4,
}


# ── Memory helpers ────────────────────────────────────────────────────────────

def force_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Workspace ─────────────────────────────────────────────────────────────────

def setup_workspace(model_name: str) -> list:
    """
    Create a fresh per-model workspace and copy images into the pool.
    Workspace layout:
        experiment_workspace/<model_name>/
            pool_images/
            train_data/images/
            train_data/labels/
            models/
    """
    model_root = os.path.join(EXPERIMENT_ROOT, model_name)

    if os.path.exists(model_root):
        try:
            shutil.rmtree(model_root)
        except Exception:
            print(f"Warning: could not remove existing workspace at {model_root}.")

    os.makedirs(f"{model_root}/pool_images",       exist_ok=True)
    os.makedirs(f"{model_root}/train_data/images", exist_ok=True)
    os.makedirs(f"{model_root}/train_data/labels", exist_ok=True)

    print(f"Workspace ready: {model_root}")

    all_images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        all_images.extend(glob.glob(os.path.join(RAW_IMAGE_DIR, ext)))

    for img in all_images:
        shutil.copy(img, f"{model_root}/pool_images/")

    return glob.glob(f"{model_root}/pool_images/*.*")


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


# ── Training wrapper ──────────────────────────────────────────────────────────

def train_model(
    model_id   : str,
    model_name : str,
    yaml_path  : str,
    epochs     : int,
    round_idx  : int,
) -> str:
    """
    Train (or fine-tune) a model for one round.
    Returns the path to the best checkpoint.
    """
    batch      = MODEL_BATCH.get(model_name, DEFAULT_BATCH)
    model_root = os.path.join(EXPERIMENT_ROOT, model_name)
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


# ── Per-loop validation (inline, avoids re-loading evaluate_results) ──────────

def _quick_val(model_path: str) -> tuple[float, float]:
    """Return (mAP50, mAP50_95) for a checkpoint."""
    model   = YOLO(model_path)
    metrics = model.val(data="validation_config.yaml", verbose=False, workers=0)
    result  = (metrics.box.map50, metrics.box.map)
    del model
    force_cleanup()
    return result


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(model_name: str) -> None:
    """
    Execute the full active-distillation pipeline for one model.

    Phases
    ------
    1. Cold-start  : label INITIAL_SEED_PCT of pool with Grounding DINO → train round 0
    2. Active loops: score pool with DFLUncertaintySampler → label top batch → retrain
    3. Evaluation  : validate every checkpoint, save CSVs and plots
    """
    model_id   = MODEL_REGISTRY[model_name]
    model_root = os.path.join(EXPERIMENT_ROOT, model_name)

    force_cleanup()
    pool_images = setup_workspace(model_name)
    yaml_path   = create_yaml(f"{model_root}/train_data")

    timer = notify_start(model_name, "Active Distillation", len(pool_images))

    teacher = TeacherLabeler(
        model_config_path = GD_CONFIG,
        model_weights_path = GD_WEIGHTS,
        class_map          = CLASS_MAP,
        output_folder      = f"{model_root}/train_data",
    )

    try:
        # ── Phase 1: Cold start ───────────────────────────────────────────────
        num_seed   = max(1, int(len(pool_images) * INITIAL_SEED_PCT))
        seed_batch = random.sample(pool_images, num_seed)

        print(f"\n--- Cold Start ({num_seed} images) [{model_name}] ---")
        notify_phase(model_name, "Cold start — labelling", f"{num_seed} images", timer)

        teacher.label_batch(seed_batch)
        force_cleanup()

        for img in seed_batch:
            if os.path.exists(img):
                os.remove(img)

        # ── Round 0: train on seed ────────────────────────────────────────────
        print(f"\n--- Training Round 0 [{model_name}] ---")
        notify_phase(model_name, "Training round 0", f"epochs={COLD_START_EPOCHS}", timer)

        current_model_path = train_model(
            model_id   = model_id,
            model_name = model_name,
            yaml_path  = yaml_path,
            epochs     = COLD_START_EPOCHS,
            round_idx  = 0,
        )

        # ── Active loops ──────────────────────────────────────────────────────
        for loop_idx in range(1, MAX_LOOPS + 1):
            print(f"\n--- Active Loop {loop_idx}/{MAX_LOOPS} [{model_name}] ---")

            current_pool = glob.glob(f"{model_root}/pool_images/*.*")
            if not current_pool:
                print("Pool exhausted — stopping early.")
                break

            # Score pool with DFL-variance sampler
            notify_phase(model_name, f"Loop {loop_idx} — scoring pool",
                         f"{len(current_pool)} images", timer)

            sampler    = DFLUncertaintySampler(current_model_path)
            num_active = max(1, int(len(pool_images) * ACTIVE_BATCH_PCT))
            hard_batch = sampler.select_batch(current_pool, batch_size=num_active)
            sampler.cleanup()
            del sampler
            force_cleanup()

            # Label selected batch
            notify_phase(model_name, f"Loop {loop_idx} — labelling",
                         f"{len(hard_batch)} images", timer)
            teacher.label_batch(hard_batch)

            for img in hard_batch:
                if os.path.exists(img):
                    os.remove(img)

            # Retrain
            print(f"Retraining [{model_name}] round {loop_idx}...")
            notify_phase(model_name, f"Loop {loop_idx} — retraining",
                         f"epochs={ACTIVE_TRAIN_EPOCHS}", timer)

            current_model_path = train_model(
                model_id   = current_model_path,   # fine-tune from previous round
                model_name = model_name,
                yaml_path  = yaml_path,
                epochs     = ACTIVE_TRAIN_EPOCHS,
                round_idx  = loop_idx,
            )

            # Quick inline validation for live ntfy updates
            try:
                map50, map50_95 = _quick_val(current_model_path)
                notify_loop(model_name, loop_idx, MAX_LOOPS, map50, map50_95, timer)
            except Exception as val_err:
                print(f"Inline validation failed: {val_err}")

        # ── Evaluation ────────────────────────────────────────────────────────
        notify_phase(model_name, "Final evaluation", "", timer)
        df = evaluate_results.evaluate(model_name=model_name)

        # Report best round
        if not df.empty:
            best_row   = df.loc[df["mAP_50"].idxmax()]
            best_map50 = float(best_row["mAP_50"])
            best_round = int(best_row["Round"])
            notify_done(model_name, best_map50, best_round, timer)

    except Exception as e:
        notify_fail(model_name, "pipeline", e, timer)
        raise


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Active distillation experiment runner.")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--model",
        choices = list(MODEL_REGISTRY.keys()),
        default = "yolov8n",
        help    = "Model to run (default: yolov8n).",
    )
    group.add_argument(
        "--all",
        action  = "store_true",
        help    = "Run all models in MODEL_REGISTRY sequentially.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.all:
        models = list(MODEL_REGISTRY.keys())
        print(f"Running all models: {models}")
        for model_name in models:
            print(f"\n{'='*60}")
            print(f"  MODEL: {model_name}")
            print(f"{'='*60}")
            run_pipeline(model_name)
    else:
        run_pipeline(args.model)


if __name__ == "__main__":
    main()
