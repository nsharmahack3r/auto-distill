"""
nguyen_sampler.py — Implements both active learning samplers from:

    Nguyen & Nguyen (2025). "A Model-Agnostic Active Learning Approach
    for Animal Detection from Camera Traps." arXiv:2507.06537.

Two samplers are provided, matching the paper exactly:

    NguenMethod1Sampler  — Diversity-driven uncertainty (Algorithm 1)
        K-means clusters the unlabelled pool into B groups using VGG-16
        image embeddings, then picks the highest-uncertainty image from
        each cluster. Uncertainty = mean(1 - detection_score) over all
        detections in an image (Eq. 1 in paper).

    NguenMethod2Sampler  — Uncertainty-driven diversity (Algorithm 2)
        Greedy selection: starts with the highest-uncertainty image, then
        iteratively picks the image that maximises a combined score
        z(x) = (1-alpha)*u(x) + alpha*v(x, X_hat)  (Eq. 5).
        Alpha decays adaptively per Eq. 6.

Key design choices to match the paper:
  - VGG-16 pretrained on ImageNet used as the task-agnostic encoder (g).
  - Image embeddings are the 4096-d vector from VGG-16's penultimate fc layer.
  - Diversity v(x, X') = mean Euclidean distance to set X' (Eq. 2).
  - Uncertainty u(x) = mean(1 - s_i) across all detections (Eq. 1).
    If no detections → u(x) = 0.5.
  - YOLOv8 used in black-box mode (outputs only, no internal access).

Dependencies: ultralytics, torchvision, torch, scikit-learn, numpy, tqdm
"""

import gc
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO
from sklearn.cluster import KMeans
from tqdm import tqdm


# ── VGG-16 encoder ────────────────────────────────────────────────────────────

_VGG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class _VGGEncoder:
    """
    Wraps torchvision VGG-16 and extracts 4096-d penultimate fc embeddings.
    Matches the encoder described in Section 3 / 4.2 of Nguyen et al.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print("[NguenSampler] Loading VGG-16 encoder (ImageNet pretrained)...")
        import os
        weight_path = "vgg16-397923af.pth"
        if os.path.exists(weight_path):
            print(f"  [VGGEncoder] Loading weights from local file {weight_path}")
            vgg = models.vgg16(weights=None)
            vgg.load_state_dict(torch.load(weight_path))
        else:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # Drop final softmax classifier; keep up to the 4096-d ReLU layer
        vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])
        vgg.eval()
        self.model = vgg.to(self.device)

    @torch.no_grad()
    def encode(self, image_path: str) -> np.ndarray:
        """Return a (4096,) float32 embedding for one image."""
        try:
            img = Image.open(image_path).convert("RGB")
            tensor = _VGG_TRANSFORM(img).unsqueeze(0).to(self.device)
            feat = self.model(tensor)
            return feat.squeeze().cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"  [VGGEncoder] Failed on {image_path}: {e}")
            return np.zeros(4096, dtype=np.float32)

    def encode_batch(self, image_paths: list) -> np.ndarray:
        """Return (N, 4096) embedding matrix."""
        return np.stack([
            self.encode(p)
            for p in tqdm(image_paths, desc="  VGG-16 embeddings", unit="img")
        ])

    def cleanup(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# ── Uncertainty scoring — Eq. 1 ──────────────────────────────────────────────

def _compute_uncertainty(model: YOLO, image_paths: list) -> np.ndarray:
    """
    u(x) = mean_i(1 - s_i(x))   where s_i is the detection score.
    If |D_f(x)| == 0  →  u(x) = 0.5  (model saw nothing; ambiguous).
    Eq. 1, Nguyen & Nguyen (2025).
    """
    scores = []
    for img_path in tqdm(image_paths, desc="  Uncertainty scores", unit="img"):
        try:
            results = model(img_path, verbose=False, conf=0.1)
            boxes = results[0].boxes
            if len(boxes) == 0:
                scores.append(0.5)
            else:
                confs = boxes.conf.cpu().numpy()
                scores.append(float(np.mean(1.0 - confs)))
        except Exception as e:
            print(f"  [Uncertainty] Error on {img_path}: {e}")
            scores.append(0.5)
    return np.array(scores, dtype=np.float32)


# ── Method 1: Diversity-driven uncertainty (Algorithm 1) ─────────────────────

class NguenMethod1Sampler:
    """
    Diversity-driven uncertainty-based active learning (Algorithm 1).

    1. Compute u(x) for all unlabelled images via YOLOv8 outputs.
    2. Extract VGG-16 embeddings.
    3. K-means cluster into B clusters.
    4. From each cluster, pick the image with the highest u(x).
    """

    def __init__(self, student_model_path: str):
        self.model = YOLO(student_model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def select_batch(self, image_list: list, batch_size: int) -> list:
        print(f"\n[Nguyen Method 1] Pool={len(image_list)}, budget={batch_size}")

        if len(image_list) <= batch_size:
            return image_list

        # Uncertainty (model outputs only)
        u_scores = _compute_uncertainty(self.model, image_list)

        # VGG-16 embeddings
        encoder = _VGGEncoder(self.device)
        embeddings = encoder.encode_batch(image_list)
        encoder.cleanup()

        # K-means → B clusters
        print(f"  K-means clustering K={batch_size}...")
        effective_k = min(batch_size, len(image_list))
        km = KMeans(n_clusters=effective_k, random_state=42, n_init="auto")
        labels = km.fit_predict(embeddings)

        # One representative per cluster: argmax u(x) within cluster
        selected = []
        for k in range(effective_k):
            idx_in_cluster = np.where(labels == k)[0]
            if len(idx_in_cluster) == 0:
                continue
            best = idx_in_cluster[np.argmax(u_scores[idx_in_cluster])]
            selected.append(image_list[best])

        # Pad if any cluster was empty
        if len(selected) < batch_size:
            chosen = set(selected)
            extras = sorted(
                [(image_list[i], u_scores[i]) for i in range(len(image_list))
                 if image_list[i] not in chosen],
                key=lambda x: x[1], reverse=True
            )
            for path, _ in extras[:batch_size - len(selected)]:
                selected.append(path)

        print(f"[Nguyen Method 1] Selected {len(selected)} images.")
        return selected

    def cleanup(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# ── Method 2: Uncertainty-driven diversity (Algorithm 2) ─────────────────────

class NguenMethod2Sampler:
    """
    Uncertainty-driven diversity-based active learning (Algorithm 2).

    Greedy selection:
      - x_1 = argmax u(x)
      - x_k = argmax z(x | X_hat)  for k = 2..B
                z = (1-alpha)*u_norm(x) + alpha*v_norm(x, X_hat)
      - alpha updated per Eq. 6:
                alpha^(i) = alpha^(i-1) - B / (2 * |unlabelled|)
    """

    def __init__(self, student_model_path: str):
        self.model = YOLO(student_model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def select_batch(
        self,
        image_list: list,
        batch_size: int,
        alpha_0: float = 0.5,
    ) -> list:
        print(f"\n[Nguyen Method 2] Pool={len(image_list)}, budget={batch_size}")

        if len(image_list) <= batch_size:
            return image_list

        u_scores = _compute_uncertainty(self.model, image_list)

        encoder = _VGGEncoder(self.device)
        embeddings = encoder.encode_batch(image_list)   # (N, 4096)
        encoder.cleanup()

        n_total = len(image_list)
        selected_indices = []
        remaining_mask = np.ones(n_total, dtype=bool)

        # First sample: highest uncertainty (Algorithm 2, line 10)
        first = int(np.argmax(u_scores))
        selected_indices.append(first)
        remaining_mask[first] = False

        alpha = alpha_0

        for k in tqdm(range(1, batch_size), desc="  Greedy selection", unit="step"):
            # Adaptive alpha update — Eq. 6
            n_unlabelled = int(remaining_mask.sum())
            if n_unlabelled > 0:
                alpha = max(0.0, alpha - batch_size / (2.0 * n_unlabelled))

            sel_embs = embeddings[selected_indices]         # (k, 4096)
            rem_idx  = np.where(remaining_mask)[0]          # (M,)
            rem_embs = embeddings[rem_idx]                  # (M, 4096)

            # Pairwise Euclidean distances (M, k) → mean diversity per image
            diffs     = rem_embs[:, np.newaxis, :] - sel_embs[np.newaxis, :, :]
            dists     = np.linalg.norm(diffs, axis=2)       # (M, k)
            diversity = dists.mean(axis=1)                  # (M,)

            # Normalise u and v to [0,1] for stable weighting
            u_rem  = u_scores[rem_idx]
            u_norm = (u_rem  - u_rem.min())  / (u_rem.max()  - u_rem.min()  + 1e-8)
            d_norm = (diversity - diversity.min()) / (diversity.max() - diversity.min() + 1e-8)

            z = (1.0 - alpha) * u_norm + alpha * d_norm
            best_local  = int(np.argmax(z))
            best_global = rem_idx[best_local]

            selected_indices.append(best_global)
            remaining_mask[best_global] = False

        selected = [image_list[i] for i in selected_indices]
        print(f"[Nguyen Method 2] Selected {len(selected)} images.")
        return selected

    def cleanup(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
