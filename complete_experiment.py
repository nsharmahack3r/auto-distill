"""
complete_experiment.py — Single entry point for all IEEE paper experiments.

Runs every experiment needed for the paper in a defined sequence, writes all
results to a timestamped results directory, and produces a master manifest
that maps every output file back to the experiment that created it.

╔══════════════════════════════════════════════════════════════════════════╗
║  EXPERIMENT PLAN                                                         ║
╠══════════════════════════════════════════════════════════════════════════╣
║  EXP-A  Pipeline comparison                                              ║
║         Pipeline 1 (static, annotate-all) vs Pipeline 2 (DFL active)    ║
║         → primary headline result for abstract and Table I               ║
║                                                                          ║
║  EXP-B  Loop ablation                                                    ║
║         DFL-variance sampler, default budget, 6 rounds                   ║
║         → marginal gain per round, optimal loop count                    ║
║                                                                          ║
║  EXP-C  Query strategy comparison                                        ║
║         DFL-variance vs random vs least-confidence, 6 rounds             ║
║         → sampler head-to-head                                           ║
║                                                                          ║
║  EXP-D  Budget sensitivity                                               ║
║         DFL-variance at 400 / 800 / 1200 image budgets                  ║
║         → annotation cost vs accuracy trade-off                          ║
║                                                                          ║
║  EXP-E  Nguyen et al. comparison (Method 1 + Method 2)                   ║
║         Both Nguyen samplers with GDINO as oracle, full 3-class dataset  ║
║         → direct baseline comparison against closest related work        ║
╚══════════════════════════════════════════════════════════════════════════╝

Output layout
─────────────
paper_results/
    <RUN_ID>/                        ← timestamped run folder
        manifest.json                ← maps every file to its experiment
        run_config.json              ← full hyperparameter snapshot
        EXP_A_pipeline/
            pipeline1_results.csv
            pipeline2_results.csv
            pipeline_comparison.csv
            pipeline_comparison_<ts>.png
            time_comparison.csv
        EXP_B_ablation/
            ablation_results.csv
            ablation_curve_<ts>.png
        EXP_C_strategy/
            strategy_results.csv
            strategy_comparison_<ts>.png
        EXP_D_budget/
            budget_results.csv
            budget_sensitivity_<ts>.png
        EXP_E_nguyen/
            nguyen_method1_results.csv
            nguyen_method2_results.csv
            nguyen_combined_results.csv
            nguyen_comparison_<ts>.png
        master_results.csv           ← all experiments merged, one row per
                                        (experiment, method/config, round)

Usage
─────
    # Run all experiments (full paper):
    python complete_experiment.py

    # Run specific experiments only:
    python complete_experiment.py --experiments A B C

    # Preview what would run without training:
    python complete_experiment.py --dry-run

    # Resume — skip experiments whose output CSVs already exist:
    python complete_experiment.py --resume

    # Use a specific run directory (e.g. to resume a previous run):
    python complete_experiment.py --resume --run-id 20260325_143000
"""

import os
import gc
import glob
import json
import shutil
import random
import argparse
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch
import pandas as pd
from ultralytics import YOLO
from gpu_config import configure_gpu

# ── Your existing modules (unchanged) ────────────────────────────────────────
from novelity_sampler   import DFLUncertaintySampler
from active_sampler     import UncertaintySampler
from teacher_labeler    import TeacherLabeler
from nguyen_sampler     import NguenMethod1Sampler, NguenMethod2Sampler
from dataset            import DatasetInfo
import evaluate_experiments
import evaluate_nguyen
from notify import (
    Timer, notify_start, notify_phase, notify_loop, notify_done, notify_fail,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these to match your environment
# ══════════════════════════════════════════════════════════════════════════════

CFG = {
    # Paths
    "raw_image_dir"     : DatasetInfo.imagesPath,
    "raw_labels_dir"    : DatasetInfo.labelsPath,
    "gd_config"         : "groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "gd_weights"        : "groundingdino_swint_ogc.pth",
    "validation_yaml"   : "validation_config.yaml",
    "workspace_root"    : "experiment_workspace",
    "nguyen_root"       : "nguyen_workspace",
    "results_root"      : "paper_results",

    # Student model
    "student_model"     : "yolov8n.pt",
    "student_name"      : "yolov8n",
    "batch_size"        : 16,

    # Active learning
    "seed_pct"          : 0.10,   # 10% cold-start
    "active_batch_pct"  : 0.05,   # 5% per loop  (Nguyen et al. protocol)
    "max_loops"         : 6,      # rounds 1-6
    "cold_start_epochs" : 10,
    "active_epochs"     : 20,

    # Budget experiment pool sizes
    "budget_levels"     : [400, 800, 1200],

    # Class map
    "class_map"         : {"rhino": 0, "zebra": 1, "leopard": 2},

    # Reproducibility
    "seed"              : 42,
}

ALL_EXPERIMENTS = ["A", "B", "C", "D", "E"]

# ══════════════════════════════════════════════════════════════════════════════
# RESULT DIRECTORY SETUP
# ══════════════════════════════════════════════════════════════════════════════

def make_run_dir(run_id: str | None = None) -> tuple[str, str]:
    """
    Create the timestamped results directory tree.
    Returns (run_dir, run_id).
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = os.path.join(CFG["results_root"], run_id)

    for subdir in [
        "EXP_A_pipeline",
        "EXP_B_ablation",
        "EXP_C_strategy",
        "EXP_D_budget",
        "EXP_E_nguyen",
    ]:
        os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)

    return run_dir, run_id


def save_run_config(run_dir: str, run_id: str, experiments: list[str]) -> None:
    """Snapshot the full configuration to run_config.json."""
    config_snap = {
        "run_id"       : run_id,
        "timestamp"    : datetime.now().isoformat(),
        "experiments"  : experiments,
        "config"       : {k: v for k, v in CFG.items() if k != "class_map"},
        "class_map"    : CFG["class_map"],
    }
    path = os.path.join(run_dir, "run_config.json")
    with open(path, "w") as f:
        json.dump(config_snap, f, indent=2)
    print(f"Config saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MANIFEST — tracks every output file
# ══════════════════════════════════════════════════════════════════════════════

class Manifest:
    """
    Builds manifest.json — a human- and machine-readable index of every
    output file, which experiment produced it, and when.
    """

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.path    = os.path.join(run_dir, "manifest.json")
        self.entries: list[dict] = []

    def record(
        self,
        experiment : str,
        file_path  : str,
        file_type  : str,
        description: str,
        status     : str = "ok",
    ) -> None:
        self.entries.append({
            "experiment"  : experiment,
            "file"        : os.path.relpath(file_path, self.run_dir),
            "type"        : file_type,          # csv | png | json
            "description" : description,
            "status"      : status,             # ok | skipped | failed
            "recorded_at" : datetime.now().isoformat(),
        })
        self._flush()

    def _flush(self) -> None:
        with open(self.path, "w") as f:
            json.dump({"run_dir": self.run_dir, "files": self.entries}, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _setup_pool(workspace_tag: str, image_budget: int | None = None) -> list:
    """
    Create a fresh workspace under experiment_workspace/<workspace_tag>/
    and copy images from the raw dataset into pool_images/.
    Returns the list of image paths in the pool.
    """
    root = os.path.join(CFG["workspace_root"], workspace_tag)
    if os.path.exists(root):
        try:
            shutil.rmtree(root)
        except Exception:
            print(f"  Warning: could not remove {root}")

    os.makedirs(f"{root}/pool_images",       exist_ok=True)
    os.makedirs(f"{root}/train_data/images", exist_ok=True)
    os.makedirs(f"{root}/train_data/labels", exist_ok=True)

    all_images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        all_images.extend(glob.glob(os.path.join(CFG["raw_image_dir"], ext)))

    if image_budget and image_budget < len(all_images):
        all_images = random.sample(all_images, image_budget)

    for img in all_images:
        shutil.copy(img, f"{root}/pool_images/")

    pool = glob.glob(f"{root}/pool_images/*.*")
    print(f"  Pool ready: {root} ({len(pool)} images)")
    return pool


def _nguyen_setup_pool(workspace_tag: str, image_budget: int | None = None) -> list:
    """Same as _setup_pool but under nguyen_workspace/."""
    root = os.path.join(CFG["nguyen_root"], workspace_tag)
    if os.path.exists(root):
        try:
            shutil.rmtree(root)
        except Exception:
            print(f"  Warning: could not remove {root}")

    os.makedirs(f"{root}/pool_images",       exist_ok=True)
    os.makedirs(f"{root}/train_data/images", exist_ok=True)
    os.makedirs(f"{root}/train_data/labels", exist_ok=True)

    all_images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        all_images.extend(glob.glob(os.path.join(CFG["raw_image_dir"], ext)))

    if image_budget and image_budget < len(all_images):
        all_images = random.sample(all_images, image_budget)

    for img in all_images:
        shutil.copy(img, f"{root}/pool_images/")

    pool = glob.glob(f"{root}/pool_images/*.*")
    print(f"  Nguyen pool ready: {root} ({len(pool)} images)")
    return pool


def _make_yaml(train_folder: str) -> str:
    abs_path = os.path.abspath(train_folder).replace("\\", "/")
    content  = (
        f"path: {abs_path}\ntrain: images\nval: images\n"
        f"names:\n  0: rhino\n  1: zebra\n  2: leopard\n"
    )
    yaml_path = os.path.join(train_folder, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(content)
    return os.path.abspath(yaml_path)


def _train(model_id: str, workspace_tag: str, yaml_path: str,
           epochs: int, round_idx: int,
           workspace_root: str | None = None) -> str:
    """Train/fine-tune student. Returns best.pt path."""
    root     = os.path.join(workspace_root or CFG["workspace_root"], workspace_tag)
    run_name = f"student_round_{round_idx}"
    project  = os.path.abspath(f"{root}/models")

    model = YOLO(model_id)
    model.train(
        data    = yaml_path,
        epochs  = epochs,
        imgsz   = 640,
        batch   = CFG["batch_size"],
        workers = 2,
        project = project,
        name    = run_name,
        verbose = False,
        amp     = True,
    )
    best_pt = os.path.join(project, run_name, "weights", "best.pt")
    del model
    _cleanup()
    return best_pt


def _quick_val(model_path: str) -> tuple[float, float]:
    model   = YOLO(model_path)
    metrics = model.val(data=CFG["validation_yaml"], verbose=False, workers=0, half=True)
    result  = (metrics.box.map50, metrics.box.map)
    del model
    _cleanup()
    return result


def _full_val(model_path: str, label: str, round_idx: int) -> dict | None:
    """Full validation including per-class mAP."""
    try:
        model   = YOLO(model_path)
        metrics = model.val(data=CFG["validation_yaml"], verbose=False, workers=0, half=True)

        row = {
            "label"      : label,
            "round"      : round_idx,
            "mAP_50"     : metrics.box.map50,
            "mAP_50_95"  : metrics.box.map,
            "model_path" : model_path,
        }

        if hasattr(metrics.box, "maps") and metrics.box.maps is not None:
            class_names = list(CFG["class_map"].keys())
            n_model     = len(metrics.box.maps)
            for i, name in enumerate(class_names):
                row[f"mAP50_{name}"] = float(metrics.box.maps[i]) if i < n_model else None

        del model
        _cleanup()
        return row

    except Exception as e:
        print(f"  Validation failed for {label} round {round_idx}: {e}")
        return None


def _sort_key(path: str) -> int:
    try:
        return int(path.split("student_round_")[1].split("\\")[0].split("/")[0])
    except (IndexError, ValueError):
        return -1


def _eval_all_checkpoints(workspace_tag: str, label: str,
                           workspace_root: str | None = None) -> pd.DataFrame:
    """Evaluate every checkpoint in a workspace. Returns DataFrame."""
    pattern = os.path.join(
        workspace_root or CFG["workspace_root"],
        workspace_tag, "models", "student_round_*", "weights", "best.pt"
    )
    paths = sorted(glob.glob(pattern), key=_sort_key)
    rows  = []
    for p in paths:
        ridx   = _sort_key(p)
        result = _full_val(p, label, ridx)
        if result:
            rows.append(result)
            print(f"    Round {ridx}: mAP@50={result['mAP_50']:.4f}  "
                  f"mAP@50-95={result['mAP_50_95']:.4f}")
    return pd.DataFrame(rows)


def _make_teacher(output_folder: str) -> TeacherLabeler:
    return TeacherLabeler(
        model_config_path  = CFG["gd_config"],
        model_weights_path = CFG["gd_weights"],
        class_map          = CFG["class_map"],
        output_folder      = output_folder,
        labels_dir         = CFG["raw_labels_dir"],
    )


def _already_done(csv_path: str, resume: bool) -> bool:
    """Return True if we should skip this experiment (resume mode + file exists)."""
    if resume and os.path.exists(csv_path):
        print(f"  SKIP (resume mode, file exists): {csv_path}")
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# ACTIVE DISTILLATION PIPELINE  (parameterised core used by EXP B/C/D/E)
# ══════════════════════════════════════════════════════════════════════════════

class _RandomSampler:
    def __init__(self, model_path=None): pass
    def select_batch(self, image_list, batch_size):
        return random.sample(image_list, min(batch_size, len(image_list)))
    def cleanup(self): pass


def _get_sampler(strategy: str, model_path: str):
    return {
        "dfl_variance"    : DFLUncertaintySampler,
        "random"          : _RandomSampler,
        "least_confidence": UncertaintySampler,
        "nguyen_method1"  : NguenMethod1Sampler,
        "nguyen_method2"  : NguenMethod2Sampler,
    }[strategy](model_path)


def _run_active_pipeline(
    workspace_tag  : str,
    strategy       : str,
    image_budget   : int | None = None,
    max_loops      : int | None = None,
    workspace_root : str | None = None,
    label          : str | None = None,
) -> pd.DataFrame:
    """
    Core active-distillation loop.  Used by all experiments that involve
    iterative annotation + training.

    Parameters
    ----------
    workspace_tag  : unique name — determines workspace folder
    strategy       : sampler key (dfl_variance / random / least_confidence /
                                   nguyen_method1 / nguyen_method2)
    image_budget   : cap on pool size (None = full dataset)
    max_loops      : active loops after cold-start (None = CFG default)
    workspace_root : override workspace root dir (for Nguyen experiments)
    label          : human-readable label for result rows

    Returns
    -------
    pd.DataFrame with per-round evaluation results.
    """
    max_loops      = max_loops      or CFG["max_loops"]
    workspace_root = workspace_root or CFG["workspace_root"]
    label          = label          or workspace_tag

    root = os.path.join(workspace_root, workspace_tag)
    _cleanup()

    pool_images = (
        _nguyen_setup_pool(workspace_tag, image_budget)
        if workspace_root == CFG["nguyen_root"]
        else _setup_pool(workspace_tag, image_budget)
    )
    yaml_path = _make_yaml(f"{root}/train_data")
    teacher   = _make_teacher(f"{root}/train_data")
    timer     = notify_start(workspace_tag, strategy, len(pool_images))

    try:
        # ── Cold start ────────────────────────────────────────────────────
        num_seed   = max(1, int(len(pool_images) * CFG["seed_pct"]))
        seed_batch = random.sample(pool_images, num_seed)

        print(f"  Cold start: labelling {num_seed} images...")
        notify_phase(workspace_tag, "Cold start", f"{num_seed} images", timer)
        teacher.label_batch(seed_batch)
        _cleanup()
        for img in seed_batch:
            if os.path.exists(img): os.remove(img)

        # ── Round 0 ───────────────────────────────────────────────────────
        print(f"  Training round 0 (epochs={CFG['cold_start_epochs']})...")
        current_model = _train(
            CFG["student_model"], workspace_tag, yaml_path,
            CFG["cold_start_epochs"], 0, workspace_root,
        )

        # ── Active loops ──────────────────────────────────────────────────
        for loop in range(1, max_loops + 1):
            current_pool = glob.glob(f"{root}/pool_images/*.*")
            if not current_pool:
                print("  Pool exhausted — stopping early.")
                break

            num_active = max(1, int(len(pool_images) * CFG["active_batch_pct"]))
            print(f"  Loop {loop}/{max_loops}: scoring {len(current_pool)} images "
                  f"(strategy={strategy})...")

            notify_phase(workspace_tag, f"Loop {loop} scoring", strategy, timer)
            sampler  = _get_sampler(strategy, current_model)
            selected = sampler.select_batch(current_pool, batch_size=num_active)
            sampler.cleanup()
            del sampler
            _cleanup()

            notify_phase(workspace_tag, f"Loop {loop} labelling",
                         f"{len(selected)} images", timer)
            teacher.label_batch(selected)
            for img in selected:
                if os.path.exists(img): os.remove(img)

            print(f"  Retraining round {loop} (epochs={CFG['active_epochs']})...")
            notify_phase(workspace_tag, f"Loop {loop} training", "", timer)
            current_model = _train(
                current_model, workspace_tag, yaml_path,
                CFG["active_epochs"], loop, workspace_root,
            )

            try:
                m50, m95 = _quick_val(current_model)
                notify_loop(workspace_tag, loop, max_loops, m50, m95, timer)
            except Exception as e:
                print(f"  Inline val failed: {e}")

        # ── Evaluate all checkpoints ──────────────────────────────────────
        print(f"  Evaluating all checkpoints for {workspace_tag}...")
        df = _eval_all_checkpoints(workspace_tag, label, workspace_root)

        if not df.empty:
            best = df.loc[df["mAP_50"].idxmax()]
            notify_done(workspace_tag, float(best["mAP_50"]),
                        int(best["round"]), timer)
        return df

    except Exception as e:
        notify_fail(workspace_tag, strategy, e, timer)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# EXP-A  Pipeline comparison  (Pipeline 1 vs Pipeline 2)
# ══════════════════════════════════════════════════════════════════════════════

def run_exp_A(run_dir: str, manifest: Manifest, resume: bool, dry_run: bool) -> pd.DataFrame | None:
    """
    EXP-A: Pipeline 1 (annotate-all, train-once) vs Pipeline 2 (DFL active).

    Pipeline 1 is implemented here directly — GDINO labels everything, then
    YOLOv8n trains for a single run (no active loops).  Its result is stored
    as a single-row DataFrame at round 0 for apples-to-apples comparison with
    Pipeline 2's best round.
    """
    out_dir  = os.path.join(run_dir, "EXP_A_pipeline")
    p1_csv   = os.path.join(out_dir, "pipeline1_results.csv")
    p2_csv   = os.path.join(out_dir, "pipeline2_results.csv")
    cmp_csv  = os.path.join(out_dir, "pipeline_comparison.csv")

    print("\n" + "═"*60)
    print("  EXP-A  Pipeline Comparison")
    print("═"*60)

    if dry_run:
        print("  [DRY RUN] Would run Pipeline 1 (static) and Pipeline 2 (DFL active).")
        return None

    # ── Pipeline 1: static baseline ──────────────────────────────────────
    p1_tag = "exp_A_pipeline1"

    if _already_done(p1_csv, resume):
        df_p1 = pd.read_csv(p1_csv)
    else:
        print("\n  --- Pipeline 1: Annotate-all static baseline ---")
        t0    = time.time()
        root  = os.path.join(CFG["workspace_root"], p1_tag)
        _cleanup()

        pool  = _setup_pool(p1_tag)
        yaml  = _make_yaml(f"{root}/train_data")
        teach = _make_teacher(f"{root}/train_data")

        # Label EVERYTHING upfront
        ann_start = time.time()
        teach.label_batch(pool)
        ann_time  = (time.time() - ann_start) / 60
        _cleanup()

        # Train once from scratch — no active loops
        train_start = time.time()
        best_pt = _train(CFG["student_model"], p1_tag, yaml,
                         CFG["cold_start_epochs"] + CFG["active_epochs"] * CFG["max_loops"],
                         0)
        train_time = (time.time() - train_start) / 60

        result = _full_val(best_pt, "Pipeline 1 (Static)", 0)
        df_p1  = pd.DataFrame([{
            **result,
            "annotation_time_min": round(ann_time,   2),
            "training_time_min"  : round(train_time, 2),
            "total_time_min"     : round(ann_time + train_time, 2),
            "pipeline"           : "Pipeline 1 (Static)",
            "images_annotated"   : len(pool),
        }])
        df_p1.to_csv(p1_csv, index=False)
        manifest.record("EXP-A", p1_csv, "csv", "Pipeline 1 static baseline results")
        print(f"  Pipeline 1 mAP@50 = {result['mAP_50']:.4f}  "
              f"({ann_time:.1f} min annot + {train_time:.1f} min train)")

    # ── Pipeline 2: active distillation (DFL) ────────────────────────────
    p2_tag = "exp_A_pipeline2"

    if _already_done(p2_csv, resume):
        df_p2 = pd.read_csv(p2_csv)
    else:
        print("\n  --- Pipeline 2: Active distillation (DFL-variance) ---")
        p2_start = time.time()
        df_p2    = _run_active_pipeline(
            workspace_tag = p2_tag,
            strategy      = "dfl_variance",
            label         = "Pipeline 2 (DFL Active)",
        )
        p2_total = (time.time() - p2_start) / 60
        df_p2["pipeline"] = "Pipeline 2 (DFL Active)"
        df_p2.to_csv(p2_csv, index=False)
        manifest.record("EXP-A", p2_csv, "csv", "Pipeline 2 DFL active results per round")

    # ── Build comparison summary ──────────────────────────────────────────
    p1_best = df_p1.iloc[0]
    p2_best = df_p2.loc[df_p2["mAP_50"].idxmax()].copy()

    cmp_rows = []
    for _, row in df_p1.iterrows():
        cmp_rows.append({
            "pipeline"           : "Pipeline 1 (Static)",
            "best_mAP_50"        : row["mAP_50"],
            "best_mAP_50_95"     : row["mAP_50_95"],
            "annotation_time_min": row.get("annotation_time_min", "N/A"),
            "training_time_min"  : row.get("training_time_min",   "N/A"),
            "total_time_min"     : row.get("total_time_min",       "N/A"),
            "images_annotated"   : row.get("images_annotated",    "N/A"),
        })
    cmp_rows.append({
        "pipeline"           : "Pipeline 2 (DFL Active)",
        "best_mAP_50"        : p2_best["mAP_50"],
        "best_mAP_50_95"     : p2_best["mAP_50_95"],
        "annotation_time_min": "see pipeline2_results.csv",
        "training_time_min"  : "see pipeline2_results.csv",
        "total_time_min"     : "see pipeline2_results.csv",
        "images_annotated"   : f"~{int(len(glob.glob(os.path.join(CFG['raw_image_dir'],'*'))) * (CFG['seed_pct'] + CFG['active_batch_pct'] * CFG['max_loops']))} (selective)",
    })
    df_cmp = pd.DataFrame(cmp_rows)
    df_cmp.to_csv(cmp_csv, index=False)
    manifest.record("EXP-A", cmp_csv, "csv", "Pipeline 1 vs Pipeline 2 summary comparison")

    # ── Plot ──────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        with mpl.rc_context(evaluate_experiments.STYLE):
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

            # Left: mAP bar chart
            ax = axes[0]
            metrics   = ["mAP@50", "mAP@50-95"]
            p1_vals   = [p1_best["mAP_50"], p1_best["mAP_50_95"]]
            p2_vals   = [p2_best["mAP_50"], p2_best["mAP_50_95"]]
            x         = range(len(metrics))
            width     = 0.35
            bars1 = ax.bar([i - width/2 for i in x], p1_vals, width,
                           label="Pipeline 1 (Static)", color="#6B7280")
            bars2 = ax.bar([i + width/2 for i in x], p2_vals, width,
                           label="Pipeline 2 (DFL Active)", color="#2563EB")
            for bar in list(bars1) + list(bars2):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
            ax.set_xticks(list(x))
            ax.set_xticklabels(metrics)
            ax.set_ylabel("Mean Average Precision")
            ax.set_title("Detection Accuracy Comparison")
            ax.legend(fontsize=8)
            ax.set_ylim(0, 1.1)

            # Right: Pipeline 2 round curve
            ax2 = axes[1]
            df_p2_sorted = df_p2.sort_values("round")
            ax2.plot(df_p2_sorted["round"], df_p2_sorted["mAP_50"],
                     marker="o", color="#2563EB", label="mAP@50")
            ax2.plot(df_p2_sorted["round"], df_p2_sorted["mAP_50_95"],
                     marker="s", linestyle="--", color="#7C3AED",
                     alpha=0.85, label="mAP@50-95")
            ax2.axhline(p1_best["mAP_50"], color="#6B7280", linestyle=":",
                        linewidth=1.5, label="Pipeline 1 baseline")
            ax2.set_xlabel("Active Learning Round")
            ax2.set_ylabel("mAP")
            ax2.set_title("Pipeline 2 Progress vs Baseline")
            ax2.legend(fontsize=8)
            ax2.set_ylim(0, 1.05)

            fig.suptitle("EXP-A: Pipeline Comparison", fontsize=13, fontweight="bold")
            fig.tight_layout()

            ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(out_dir, f"pipeline_comparison_{ts}.png")
            fig.savefig(fname, dpi=300, bbox_inches="tight")
            plt.close(fig)
            manifest.record("EXP-A", fname, "png", "Pipeline 1 vs Pipeline 2 comparison chart")
            print(f"  EXP-A plot saved → {fname}")
    except Exception as e:
        print(f"  EXP-A plot failed: {e}")

    combined = pd.concat([df_p1, df_p2], ignore_index=True)
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# EXP-B  Loop Ablation
# ══════════════════════════════════════════════════════════════════════════════

def run_exp_B(run_dir: str, manifest: Manifest, resume: bool, dry_run: bool) -> pd.DataFrame | None:
    out_dir = os.path.join(run_dir, "EXP_B_ablation")
    csv     = os.path.join(out_dir, "ablation_results.csv")

    print("\n" + "═"*60)
    print("  EXP-B  Loop Ablation")
    print("═"*60)

    if dry_run:
        print("  [DRY RUN] DFL-variance, default budget, 6 loops.")
        return None
    if _already_done(csv, resume):
        return pd.read_csv(csv)

    df = _run_active_pipeline("exp_B_ablation", "dfl_variance",
                              label="DFL Ablation")
    df["experiment"] = "EXP-B"
    df.to_csv(csv, index=False)
    manifest.record("EXP-B", csv, "csv", "Loop ablation mAP per round")

    try:
        fname = evaluate_experiments.plot_ablation(df.rename(columns={"round": "Round",
                                                                        "mAP_50": "mAP_50",
                                                                        "mAP_50_95": "mAP_50_95"}),
                                                    out_dir)
        manifest.record("EXP-B", fname, "png", "Loop ablation curve")
    except Exception as e:
        print(f"  EXP-B plot failed: {e}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# EXP-C  Query Strategy Comparison
# ══════════════════════════════════════════════════════════════════════════════

def run_exp_C(run_dir: str, manifest: Manifest, resume: bool, dry_run: bool) -> pd.DataFrame | None:
    out_dir    = os.path.join(run_dir, "EXP_C_strategy")
    csv        = os.path.join(out_dir, "strategy_results.csv")
    strategies = ["dfl_variance", "random", "least_confidence"]

    print("\n" + "═"*60)
    print("  EXP-C  Query Strategy Comparison")
    print("═"*60)

    if dry_run:
        print(f"  [DRY RUN] Strategies: {strategies}")
        return None
    if _already_done(csv, resume):
        return pd.read_csv(csv)

    all_dfs = []
    for strategy in strategies:
        print(f"\n  Strategy: {strategy}")
        df = _run_active_pipeline(
            f"exp_C_{strategy}", strategy,
            label=strategy.replace("_", " ").title(),
        )
        df["Strategy"]   = strategy
        df["experiment"] = "EXP-C"
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(csv, index=False)
    manifest.record("EXP-C", csv, "csv", "Strategy comparison mAP per round per strategy")

    try:
        plot_df = combined.rename(columns={"round": "Round", "mAP_50": "mAP_50"})
        fname   = evaluate_experiments.plot_strategy_comparison(plot_df, out_dir)
        manifest.record("EXP-C", fname, "png", "Strategy comparison line chart")
    except Exception as e:
        print(f"  EXP-C plot failed: {e}")

    return combined


# ══════════════════════════════════════════════════════════════════════════════
# EXP-D  Budget Sensitivity
# ══════════════════════════════════════════════════════════════════════════════

def run_exp_D(run_dir: str, manifest: Manifest, resume: bool, dry_run: bool) -> pd.DataFrame | None:
    out_dir = os.path.join(run_dir, "EXP_D_budget")
    csv     = os.path.join(out_dir, "budget_results.csv")
    budgets = CFG["budget_levels"]

    print("\n" + "═"*60)
    print("  EXP-D  Budget Sensitivity")
    print("═"*60)

    if dry_run:
        print(f"  [DRY RUN] Budget levels: {budgets}")
        return None
    if _already_done(csv, resume):
        return pd.read_csv(csv)

    all_dfs = []
    for budget in budgets:
        print(f"\n  Budget: {budget} images")
        df = _run_active_pipeline(
            f"exp_D_budget_{budget}", "dfl_variance",
            image_budget = budget,
            label        = f"Budget {budget}",
        )
        df["Budget"]     = budget
        df["experiment"] = "EXP-D"
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(csv, index=False)
    manifest.record("EXP-D", csv, "csv", "Budget sensitivity mAP per round per budget")

    try:
        plot_df = combined.rename(columns={"round": "Round", "mAP_50": "mAP_50"})
        fname   = evaluate_experiments.plot_budget_sensitivity(plot_df, out_dir)
        manifest.record("EXP-D", fname, "png", "Budget sensitivity chart")
    except Exception as e:
        print(f"  EXP-D plot failed: {e}")

    return combined


# ══════════════════════════════════════════════════════════════════════════════
# EXP-E  Nguyen et al. Comparison
# ══════════════════════════════════════════════════════════════════════════════

def run_exp_E(run_dir: str, manifest: Manifest, resume: bool, dry_run: bool) -> pd.DataFrame | None:
    out_dir  = os.path.join(run_dir, "EXP_E_nguyen")
    m1_csv   = os.path.join(out_dir, "nguyen_method1_results.csv")
    m2_csv   = os.path.join(out_dir, "nguyen_method2_results.csv")
    comb_csv = os.path.join(out_dir, "nguyen_combined_results.csv")

    print("\n" + "═"*60)
    print("  EXP-E  Nguyen et al. Comparison")
    print("═"*60)

    if dry_run:
        print("  [DRY RUN] Nguyen Method 1 + Method 2, full 3-class dataset, GDINO oracle.")
        return None

    all_dfs = []

    # ── Method 1 ─────────────────────────────────────────────────────────
    if _already_done(m1_csv, resume):
        df_m1 = pd.read_csv(m1_csv)
    else:
        print("\n  Nguyen Method 1 (Diversity-driven uncertainty)...")
        df_m1 = _run_active_pipeline(
            workspace_tag  = "method1",
            strategy       = "nguyen_method1",
            workspace_root = CFG["nguyen_root"],
            label          = "Nguyen Method 1 (Div-Unc)",
        )
        df_m1["method"]     = "method1"
        df_m1["experiment"] = "EXP-E"
        df_m1.to_csv(m1_csv, index=False)
        manifest.record("EXP-E", m1_csv, "csv",
                         "Nguyen Method 1 mAP per round (full 3-class dataset)")

    all_dfs.append(df_m1)

    # ── Method 2 ─────────────────────────────────────────────────────────
    if _already_done(m2_csv, resume):
        df_m2 = pd.read_csv(m2_csv)
    else:
        print("\n  Nguyen Method 2 (Uncertainty-driven diversity)...")
        df_m2 = _run_active_pipeline(
            workspace_tag  = "method2",
            strategy       = "nguyen_method2",
            workspace_root = CFG["nguyen_root"],
            label          = "Nguyen Method 2 (Unc-Div)",
        )
        df_m2["method"]     = "method2"
        df_m2["experiment"] = "EXP-E"
        df_m2.to_csv(m2_csv, index=False)
        manifest.record("EXP-E", m2_csv, "csv",
                         "Nguyen Method 2 mAP per round (full 3-class dataset)")

    all_dfs.append(df_m2)

    # ── Combined output + plot ────────────────────────────────────────────
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(comb_csv, index=False)
    manifest.record("EXP-E", comb_csv, "csv",
                     "Nguyen Methods 1+2 combined results")

    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        with mpl.rc_context(evaluate_experiments.STYLE):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            colors = {
                "Nguyen Method 1 (Div-Unc)": "#2563EB",
                "Nguyen Method 2 (Unc-Div)": "#DC2626",
            }
            for label, group in combined.groupby("label"):
                group = group.sort_values("round")
                ax.plot(group["round"], group["mAP_50"],
                        marker="o", linewidth=2, label=label,
                        color=colors.get(label, "gray"))

            ax.set_xlabel("Active Learning Round")
            ax.set_ylabel("mAP@50")
            ax.set_title("EXP-E: Nguyen et al. Methods — mAP@50 per Round\n"
                         "(Full 3-class dataset, GDINO oracle)")
            ax.legend(fontsize=9)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.35, linestyle="--")

            ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(out_dir, f"nguyen_comparison_{ts}.png")
            fig.savefig(fname, dpi=300, bbox_inches="tight")
            plt.close(fig)
            manifest.record("EXP-E", fname, "png",
                             "Nguyen Method 1 vs Method 2 mAP@50 chart")
            print(f"  EXP-E plot saved → {fname}")
    except Exception as e:
        print(f"  EXP-E plot failed: {e}")

    return combined


# ══════════════════════════════════════════════════════════════════════════════
# MASTER RESULTS CSV
# ══════════════════════════════════════════════════════════════════════════════

def build_master_csv(run_dir: str, results: dict, manifest: Manifest) -> None:
    """
    Merge all per-experiment DataFrames into one master_results.csv.
    Columns: experiment, label, round, mAP_50, mAP_50_95, [per-class cols...]
    """
    rows = []
    for exp_id, df in results.items():
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            entry = {
                "experiment" : exp_id,
                "label"      : row.get("label",    row.get("pipeline", "unknown")),
                "round"      : row.get("round",    row.get("Round", -1)),
                "mAP_50"     : row.get("mAP_50",   None),
                "mAP_50_95"  : row.get("mAP_50_95",None),
                "mAP50_rhino"  : row.get("mAP50_rhino",   None),
                "mAP50_zebra"  : row.get("mAP50_zebra",   None),
                "mAP50_leopard": row.get("mAP50_leopard",  None),
                "model_path"   : row.get("model_path", row.get("Model_Path", "")),
            }
            rows.append(entry)

    if not rows:
        print("  No results to merge into master CSV.")
        return

    master = pd.DataFrame(rows).sort_values(["experiment", "label", "round"])
    path   = os.path.join(run_dir, "master_results.csv")
    master.to_csv(path, index=False)
    manifest.record("ALL", path, "csv",
                     "Master results — all experiments, all rounds, all methods")
    print(f"\nMaster CSV saved → {path}")
    print(master[["experiment","label","round","mAP_50","mAP_50_95"]].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# CLI + MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all IEEE paper experiments in sequence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--experiments", nargs="+", choices=ALL_EXPERIMENTS,
        default=ALL_EXPERIMENTS,
        metavar="EXP",
        help="Which experiments to run (default: all). E.g. --experiments A B E",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run without executing any training.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip experiments whose output CSVs already exist in the run directory.",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Use a specific run ID (e.g. 20260325_143000). "
             "Useful with --resume to continue a previous run.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    CFG["seed"] = args.seed

    configure_gpu()

    run_dir, run_id = make_run_dir(args.run_id)
    manifest        = Manifest(run_dir)

    print(f"""
╔══════════════════════════════════════════════════════════╗
║          COMPLETE EXPERIMENT RUNNER                      ║
╠══════════════════════════════════════════════════════════╣
║  Run ID      : {run_id:<42}                              ║
║  Results dir : {run_dir:<42}                             ║
║  Experiments : {str(args.experiments):<42}               ║
║  Dry run     : {str(args.dry_run):<42}                   ║
║  Resume      : {str(args.resume):<42}                    ║
║  Seed        : {str(args.seed):<42}                      ║
╚══════════════════════════════════════════════════════════╝
""")

    if not args.dry_run:
        save_run_config(run_dir, run_id, args.experiments)

    # ── Dispatch ──────────────────────────────────────────────────────────
    exp_runners = {
        "A": run_exp_A,
        "B": run_exp_B,
        "C": run_exp_C,
        "D": run_exp_D,
        "E": run_exp_E,
    }

    results  = {}
    failed   = []
    wall_t0  = time.time()

    for exp_id in args.experiments:
        t0 = time.time()
        try:
            df = exp_runners[exp_id](run_dir, manifest, args.resume, args.dry_run)
            results[exp_id] = df
            elapsed = (time.time() - t0) / 60
            status  = "SKIPPED" if (args.resume and df is not None and exp_id in results) else "OK"
            print(f"\n  EXP-{exp_id} complete in {elapsed:.1f} min  [{status}]")
        except Exception as e:
            elapsed = (time.time() - t0) / 60
            print(f"\n  EXP-{exp_id} FAILED after {elapsed:.1f} min: {e}")
            traceback.print_exc()
            failed.append(exp_id)
            manifest.record(f"EXP-{exp_id}", run_dir, "error",
                             f"Experiment failed: {e}", status="failed")

    # ── Master CSV ────────────────────────────────────────────────────────
    if not args.dry_run:
        build_master_csv(run_dir, results, manifest)

    # ── Final summary ──────────────────────────────────────────────────────
    total_min = (time.time() - wall_t0) / 60
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  RUN COMPLETE                                            ║
╠══════════════════════════════════════════════════════════╣
║  Total wall time : {total_min:.1f} min{'':<36}║
║  Experiments OK  : {str([e for e in args.experiments if e not in failed]):<42}║
║  Failed          : {str(failed) if failed else 'none':<42}║
║  Results dir     : {run_dir:<42}║
╚══════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()