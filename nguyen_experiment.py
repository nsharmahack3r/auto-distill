"""
nguyen_experiment.py — Runs the Nguyen & Nguyen (2025) active learning
pipeline on your wildlife dataset for direct comparison against your
Pipeline 2 (DFL-variance active distillation).

Paper:
    Nguyen & Nguyen (2025). "A Model-Agnostic Active Learning Approach
    for Animal Detection from Camera Traps." arXiv:2507.06537.

What this script does
─────────────────────
Both methods from the paper (Algorithm 1 and Algorithm 2) are run
sequentially, each in its own isolated workspace under:

    nguyen_workspace/
        method1/
        method2/

Each run replicates the paper's experimental protocol on YOUR dataset:
  • Seed        : 10% of pool  (matching the paper's 10% initialisation)
  • Budget      : 5% per loop  (matching the paper's 5% per iteration)
  • Iterations  : 6 active loops after cold-start (matching the paper)
  • Student     : YOLOv8n  (same model used in the paper)
  • Annotation  : Grounding DINO (your teacher — replaces human labeller)
  • Evaluation  : mAP@50 and mAP@50-95 on your validation_config.yaml

Key difference from the paper:
  The paper uses a human annotator as the oracle. Here, Grounding DINO
  replaces the human oracle — this is the contribution you are testing.

Usage
─────
    # Run both methods:
    python nguyen_experiment.py

    # Run one method only:
    python nguyen_experiment.py --method 1
    python nguyen_experiment.py --method 2
"""

import os
import gc
import glob
import shutil
import random
import argparse
import torch
from ultralytics import YOLO

from nguyen_sampler import NguenMethod1Sampler, NguenMethod2Sampler
from teacher_labeler import TeacherLabeler
from dataset import DatasetInfo
import evaluate_nguyen
from notify import (
    Timer,
    notify_start,
    notify_phase,
    notify_loop,
    notify_done,
    notify_fail,
)

# ── Paths (shared with main_experiment.py) ────────────────────────────────────
RAW_IMAGE_DIR   = DatasetInfo.imagesPath
EXPERIMENT_ROOT = "nguyen_workspace"

GD_CONFIG  = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_WEIGHTS = "groundingdino_swint_ogc.pth"

# CLASS_MAP = {
#     "rhino"  : 0,
#     "zebra"  : 1,
#     "leopard": 2,
# }

CLASS_MAP = {
    # "rhino"  : 0,
    "zebra"  : 0,
    # "leopard": 2,
}


# ── Hyper-parameters (matching Nguyen et al. Section 4.2) ────────────────────
STUDENT_MODEL_ID   = "yolov8n.pt"   # same model family used in the paper
INITIAL_SEED_PCT   = 0.10           # 10% cold-start  (paper: 10%)
ACTIVE_BATCH_PCT   = 0.10           # 5% per loop      (paper: 5%)
MAX_LOOPS          = 8              # 6 active loops   (paper: 6 iterations)
COLD_START_EPOCHS  = 10             # same as your Pipeline 2
ACTIVE_TRAIN_EPOCHS = 20            # same as your Pipeline 2
DEFAULT_BATCH      = 16

# ── Memory helpers ────────────────────────────────────────────────────────────

def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Workspace setup ───────────────────────────────────────────────────────────

def setup_workspace(method_name: str) -> list:
    """
    Create a fresh per-method workspace and populate the image pool.

    Layout:
        nguyen_workspace/<method_name>/
            pool_images/
            train_data/images/
            train_data/labels/
            models/
    """
    root = os.path.join(EXPERIMENT_ROOT, method_name)

    if os.path.exists(root):
        try:
            shutil.rmtree(root)
        except Exception:
            print(f"Warning: could not remove {root}")

    os.makedirs(f"{root}/pool_images",       exist_ok=True)
    os.makedirs(f"{root}/train_data/images", exist_ok=True)
    os.makedirs(f"{root}/train_data/labels", exist_ok=True)
    os.makedirs(f"{root}/models",            exist_ok=True)

    all_images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        all_images.extend(glob.glob(os.path.join(RAW_IMAGE_DIR, ext)))

    for img in all_images:
        shutil.copy(img, f"{root}/pool_images/")

    pool = glob.glob(f"{root}/pool_images/*.*")
    print(f"Workspace ready: {root}  ({len(pool)} images in pool)")
    return pool


def create_yaml(output_folder: str) -> str:
    abs_path = os.path.abspath(output_folder).replace("\\", "/")
    content = (
        f"path: {abs_path}\n"
        f"train: images\n"
        f"val:   images\n"
        f"names:\n"
        f"  0: rhino\n"
        f"  1: zebra\n"
        f"  2: leopard\n"
    )
    yaml_path = os.path.join(output_folder, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(content)
    return os.path.abspath(yaml_path)


# ── Training wrapper ──────────────────────────────────────────────────────────

def train_student(
    model_id:    str,
    method_name: str,
    yaml_path:   str,
    epochs:      int,
    round_idx:   int,
) -> str:
    """Train or fine-tune the student. Returns path to best.pt checkpoint."""
    root     = os.path.join(EXPERIMENT_ROOT, method_name)
    run_name = f"student_round_{round_idx}"
    project  = os.path.abspath(f"{root}/models")

    model = YOLO(model_id)
    model.train(
        data    = yaml_path,
        epochs  = epochs,
        imgsz   = 640,
        batch   = DEFAULT_BATCH,
        workers = 2 if round_idx == 0 else 1,
        project = project,
        name    = run_name,
        verbose = False,
    )
    best_pt = os.path.join(project, run_name, "weights", "best.pt")
    del model
    _cleanup()
    return best_pt


def _quick_val(model_path: str) -> tuple:
    """Return (mAP50, mAP50_95) for a checkpoint."""
    model   = YOLO(model_path)
    metrics = model.val(data="validation_config.yaml", verbose=False, workers=0)
    result  = (metrics.box.map50, metrics.box.map)
    del model
    _cleanup()
    return result


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run_nguyen_method(method: int) -> None:
    """
    Execute one full Nguyen et al. pipeline run (Method 1 or Method 2).

    Phases
    ──────
    1. Cold-start : randomly seed INITIAL_SEED_PCT of pool →
                    Grounding DINO labels it → train round 0
    2. Active loops (MAX_LOOPS):
                    score pool with Nguyen sampler →
                    Grounding DINO labels selected batch →
                    fine-tune student from previous checkpoint
    3. Evaluate   : validate all checkpoints, save CSV + plots
    """
    assert method in (1, 2), "method must be 1 or 2"
    method_name = f"method{method}"

    _cleanup()
    pool_images = setup_workspace(method_name)
    root        = os.path.join(EXPERIMENT_ROOT, method_name)
    yaml_path   = create_yaml(f"{root}/train_data")
    timer       = notify_start(f"nguyen_{method_name}", "Nguyen et al.", len(pool_images))

    teacher = TeacherLabeler(
        model_config_path = GD_CONFIG,
        model_weights_path = GD_WEIGHTS,
        class_map          = CLASS_MAP,
        output_folder      = f"{root}/train_data",
    )

    try:
        # ── Phase 1: Cold start ───────────────────────────────────────────
        num_seed   = max(1, int(len(pool_images) * INITIAL_SEED_PCT))
        seed_batch = random.sample(pool_images, num_seed)

        print(f"\n--- Cold Start ({num_seed} images) [nguyen_{method_name}] ---")
        notify_phase(f"nguyen_{method_name}", "Cold start — labelling",
                     f"{num_seed} images", timer)

        teacher.label_batch(seed_batch)
        _cleanup()

        for img in seed_batch:
            if os.path.exists(img):
                os.remove(img)

        # ── Round 0: train on seed ────────────────────────────────────────
        print(f"\n--- Training Round 0 [nguyen_{method_name}] ---")
        notify_phase(f"nguyen_{method_name}", "Training round 0",
                     f"epochs={COLD_START_EPOCHS}", timer)

        current_model = train_student(
            model_id    = STUDENT_MODEL_ID,
            method_name = method_name,
            yaml_path   = yaml_path,
            epochs      = COLD_START_EPOCHS,
            round_idx   = 0,
        )

        # ── Active loops ──────────────────────────────────────────────────
        for loop in range(1, MAX_LOOPS + 1):
            print(f"\n--- Active Loop {loop}/{MAX_LOOPS} [nguyen_{method_name}] ---")

            current_pool = glob.glob(f"{root}/pool_images/*.*")
            if not current_pool:
                print("Pool exhausted — stopping early.")
                break

            num_active = max(1, int(len(pool_images) * ACTIVE_BATCH_PCT))

            # ── Select batch using Nguyen sampler ─────────────────────────
            notify_phase(f"nguyen_{method_name}", f"Loop {loop} — sampling",
                         f"{len(current_pool)} images", timer)

            if method == 1:
                sampler = NguenMethod1Sampler(current_model)
            else:
                sampler = NguenMethod2Sampler(current_model)

            selected = sampler.select_batch(current_pool, batch_size=num_active)
            sampler.cleanup()
            del sampler
            _cleanup()

            # ── Label selected batch with Grounding DINO ──────────────────
            notify_phase(f"nguyen_{method_name}", f"Loop {loop} — labelling",
                         f"{len(selected)} images", timer)

            teacher.label_batch(selected)
            for img in selected:
                if os.path.exists(img):
                    os.remove(img)

            # ── Fine-tune student ─────────────────────────────────────────
            print(f"Retraining [nguyen_{method_name}] round {loop}...")
            notify_phase(f"nguyen_{method_name}", f"Loop {loop} — retraining",
                         f"epochs={ACTIVE_TRAIN_EPOCHS}", timer)

            current_model = train_student(
                model_id    = current_model,   # fine-tune from previous
                method_name = method_name,
                yaml_path   = yaml_path,
                epochs      = ACTIVE_TRAIN_EPOCHS,
                round_idx   = loop,
            )

            # Inline validation for live progress notifications
            try:
                map50, map50_95 = _quick_val(current_model)
                notify_loop(f"nguyen_{method_name}", loop, MAX_LOOPS,
                            map50, map50_95, timer)
            except Exception as e:
                print(f"Inline validation failed: {e}")

        # ── Final evaluation ──────────────────────────────────────────────
        notify_phase(f"nguyen_{method_name}", "Final evaluation", "", timer)
        df = evaluate_nguyen.evaluate(method_name=method_name)

        if not df.empty:
            best_row   = df.loc[df["mAP_50"].idxmax()]
            notify_done(f"nguyen_{method_name}",
                        float(best_row["mAP_50"]),
                        int(best_row["Round"]),
                        timer)

    except Exception as e:
        notify_fail(f"nguyen_{method_name}", "pipeline", e, timer)
        raise


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Nguyen et al. (2025) active learning comparison experiment."
    )
    parser.add_argument(
        "--method",
        type    = int,
        choices = [1, 2],
        default = None,
        help    = "Run only Method 1 or Method 2. Omit to run both."
    )
    return parser.parse_args()


def main():
    args    = parse_args()
    methods = [args.method] if args.method else [1, 2]

    for m in methods:
        print(f"\n{'='*60}")
        print(f"  NGUYEN ET AL. METHOD {m}")
        print(f"{'='*60}")
        run_nguyen_method(m)


if __name__ == "__main__":
    main()
