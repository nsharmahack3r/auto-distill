"""
active_sampler.py — Least-confidence uncertainty sampler.

Scores each image by (1 - max_detection_confidence).
Used as a simple baseline to compare against DFLUncertaintySampler.
"""

import numpy as np
from ultralytics import YOLO
from tqdm import tqdm


class UncertaintySampler:
    """
    Selects the most uncertain images from an unlabelled pool using
    least-confidence sampling: uncertainty = 1 - max(box confidence).
    """

    def __init__(self, student_model_path: str):
        self.model = YOLO(student_model_path)

    def calculate_uncertainty(self, image_path: str) -> float:
        """
        Returns an uncertainty score in [0, 1] for a single image.

        - No detections → 0.5  (ambiguous; model saw nothing)
        - Detections present → 1 - max_confidence
          A weak best-guess (conf=0.4) gives uncertainty=0.6.
        """
        results = self.model.predict(image_path, verbose=False, conf=0.1)
        boxes   = results[0].boxes

        if len(boxes) == 0:
            return 0.5

        max_conf = float(np.max(boxes.conf.cpu().numpy()))
        return 1.0 - max_conf

    def select_batch(self, image_pool: list, batch_size: int,
                     infer_batch: int = 16) -> list:
        """
        Scan the unlabelled pool and return the top-k most uncertain paths.
        Uses batched inference for GPU efficiency.
        """
        print(f"Scanning {len(image_pool)} images "
              f"(least-confidence sampler, infer_batch={infer_batch})...")

        scores = []
        for i in tqdm(range(0, len(image_pool), infer_batch),
                      desc="  LC scoring", unit="batch"):
            chunk = image_pool[i : i + infer_batch]
            results = self.model.predict(chunk, verbose=False, conf=0.1, half=True)
            for img, res in zip(chunk, results):
                boxes = res.boxes
                if len(boxes) == 0:
                    scores.append((img, 0.5))
                else:
                    max_conf = float(np.max(boxes.conf.cpu().numpy()))
                    scores.append((img, 1.0 - max_conf))

        scores.sort(key=lambda x: x[1], reverse=True)

        print(f"  Most uncertain : {scores[0][1]:.4f}")
        print(f"  Cutoff score   : {scores[batch_size - 1][1]:.4f}")

        return [path for path, _ in scores[:batch_size]]

    def cleanup(self):
        del self.model
