"""
novelity_sampler.py — DFL-variance uncertainty sampler (novel contribution).

Hooks into YOLOv8's internal Distribution Focal Loss (DFL) box-regression
layers to extract the *variance* of the predicted box distributions, rather
than relying on the post-softmax confidence score.

High variance in a bounding-box distribution means the model is geometrically
uncertain about where the object boundary is — a stronger signal than
classification confidence alone.

Model compatibility
-------------------
YOLOv8  (n / s / m / l / x)
  → hooks cv2 layers of the Detect head  (default)

RT-DETR (l / x)
  → hooks the bbox_head linear layers of the decoder
  → falls back to confidence-based scoring if hook attachment fails

YOLOv9 / YOLOv10 / YOLOv11
  → same hook target as YOLOv8 (Detect head cv2); falls back on failure
"""

import torch
import torch.nn.functional as F
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm


class DFLUncertaintySampler:
    """
    Selects images whose bounding-box distributions show the highest
    geometric variance across all detection scales.
    """

    def __init__(self, model_path: str):
        self.model       = YOLO(model_path)
        self.device      = self.model.device
        self.activations = {}
        self.hooks       = []
        self._model_type = self._detect_model_type()
        self._hook_ok    = self._attach_hooks()

    # ── Model-type detection ──────────────────────────────────────────────────

    def _detect_model_type(self) -> str:
        """
        Returns a normalised model-family string:
        'yolo'  for YOLOv8 / v9 / v10 / v11
        'rtdetr' for RT-DETR
        """
        try:
            arch = type(self.model.model).__name__.lower()
            if "rtdetr" in arch or "detr" in arch:
                return "rtdetr"
        except Exception:
            pass

        try:
            last_layer = type(self.model.model.model[-1]).__name__.lower()
            if "detect" in last_layer or "segment" in last_layer:
                return "yolo"
        except Exception:
            pass

        return "yolo"  # safe default

    # ── Hook attachment ───────────────────────────────────────────────────────

    def _attach_hooks(self) -> bool:
        """
        Attach forward hooks to the appropriate layers.
        Returns True if hooks were attached successfully.
        """
        try:
            if self._model_type == "rtdetr":
                return self._attach_rtdetr_hooks()
            return self._attach_yolo_hooks()
        except Exception as e:
            print(f"[DFLSampler] Hook attachment failed ({e}); "
                  "falling back to confidence-based scoring.")
            return False

    def _attach_yolo_hooks(self) -> bool:
        """Hook the cv2 (box regression) layers of the YOLOv8/v9/v10/v11 Detect head."""
        head = self.model.model.model[-1]
        if not hasattr(head, "cv2"):
            raise AttributeError("No 'cv2' attribute on final layer — unexpected head structure.")

        for i, layer in enumerate(head.cv2):
            hook = layer.register_forward_hook(self._capture(f"scale_{i}"))
            self.hooks.append(hook)

        print(f"[DFLSampler] Attached {len(self.hooks)} YOLO DFL hooks.")
        return True

    def _attach_rtdetr_hooks(self) -> bool:
        """
        Hook the bounding-box prediction heads of the RT-DETR decoder.
        RT-DETR uses a set of Linear layers (bbox_head) per decoder layer
        rather than DFL bins, so we capture the raw bbox logits and compute
        their variance directly.
        """
        attached = 0
        for name, module in self.model.model.named_modules():
            if "bbox_head" in name and isinstance(module, torch.nn.Linear):
                hook = module.register_forward_hook(self._capture(f"rtdetr_{name}"))
                self.hooks.append(hook)
                attached += 1

        if attached == 0:
            raise AttributeError("No RT-DETR bbox_head Linear layers found.")

        print(f"[DFLSampler] Attached {attached} RT-DETR bbox hooks.")
        return True

    def _capture(self, name: str):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook

    # ── Variance computation ──────────────────────────────────────────────────

    def _dfl_variance(self, tensor: torch.Tensor) -> float:
        """
        Compute mean top-10% variance from a YOLOv8 DFL activation.
        Input: [B, 64, H, W]  (64 = 4 coords × 16 bins)
        """
        b, c, h, w = tensor.shape
        box_dist  = tensor.view(b, 4, 16, h, w)
        prob      = F.softmax(box_dist, dim=2)
        vals      = torch.arange(16, device=tensor.device, dtype=torch.float32).view(1, 1, 16, 1, 1)
        mean      = (prob * vals).sum(dim=2, keepdim=True)
        variance  = (prob * (vals - mean) ** 2).sum(dim=2)   # [B, 4, H, W]
        var_map   = variance.sum(dim=1)                       # [B, H, W]

        k            = max(1, int(var_map.numel() * 0.10))
        top_vals, _  = torch.topk(var_map.flatten(), k)
        return top_vals.mean().item()

    def _rtdetr_variance(self, tensor: torch.Tensor) -> float:
        """
        Compute variance of raw bbox logits from an RT-DETR bbox_head output.
        Input: [B, num_queries, 4] or [B, 4]
        """
        return tensor.float().var().item()

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _confidence_fallback(self, image_path: str) -> float:
        """Least-confidence score used when hooks are unavailable."""
        results = self.model(image_path, verbose=False, conf=0.1)
        boxes   = results[0].boxes
        if len(boxes) == 0:
            return 0.5
        return 1.0 - float(boxes.conf.cpu().numpy().max())

    def get_uncertainty_score(self, image_path: str) -> float:
        """
        Run inference and return a structural uncertainty score.
        Falls back to confidence-based scoring if hooks failed.
        """
        if not self._hook_ok:
            return self._confidence_fallback(image_path)

        self.activations = {}
        self.model(image_path, verbose=False, conf=0.1)

        total = 0.0
        for key, feat in self.activations.items():
            if key.startswith("rtdetr_"):
                total += self._rtdetr_variance(feat)
            else:
                total += self._dfl_variance(feat)

        return total

    # ── Batch selection ───────────────────────────────────────────────────────

    def select_batch(self, image_list: list, batch_size: int,
                     infer_batch: int = 16) -> list:
        """Return the top-k highest-uncertainty image paths.

        Uses batched inference for GPU efficiency — processes `infer_batch`
        images per forward pass instead of one at a time.
        """
        print(f"Scoring {len(image_list)} images "
              f"(DFL-variance, model={self._model_type}, "
              f"infer_batch={infer_batch})...")

        scores = []
        for i in tqdm(range(0, len(image_list), infer_batch),
                      desc="  DFL scoring", unit="batch"):
            chunk = image_list[i : i + infer_batch]

            if not self._hook_ok:
                # Fallback: batch predict then score by confidence
                results = self.model(chunk, verbose=False, conf=0.1, half=True)
                for img, res in zip(chunk, results):
                    boxes = res.boxes
                    if len(boxes) == 0:
                        scores.append((img, 0.5))
                    else:
                        scores.append((img, 1.0 - float(boxes.conf.cpu().numpy().max())))
            else:
                # DFL hook scoring: must run one image at a time because
                # hooks capture per-batch activations
                for img in chunk:
                    scores.append((img, self.get_uncertainty_score(img)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [path for path, _ in scores[:batch_size]]

    def cleanup(self):
        """Remove all hooks and free the model."""
        for h in self.hooks:
            h.remove()
        del self.model
