import os
import torch
import cv2
import gc
import shutil
import time
import csv
from groundingdino.util.inference import load_model, load_image, predict

class TeacherLabeler:
    def __init__(self, model_config_path, model_weights_path, class_map,
                 output_folder="dataset_active", labels_dir=None):
        """
        Parameters
        ----------
        labels_dir : str or None
            Path to a directory of pre-existing YOLO label .txt files.
            When set, label_batch() copies labels from here instead of
            running GroundingDINO inference — massively faster.
        """
        self.output_folder = output_folder
        self.class_map = class_map
        self.text_prompt = " . ".join(class_map.keys())
        
        # We store paths but DO NOT load the model in __init__ to save RAM
        self.config_path = model_config_path
        self.weights_path = model_weights_path
        self.model = None

        # Cached labels directory (bypass GDINO when available)
        self.labels_dir = labels_dir
        
        os.makedirs(f"{self.output_folder}/images", exist_ok=True)
        os.makedirs(f"{self.output_folder}/labels", exist_ok=True)

        # --- Initialize CSV Log ---
        self.csv_path = os.path.join(self.output_folder, "annotation_time_log.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Image_Name", "Time_Seconds", "Status"])

    def _load_model(self):
        """Loads model to GPU only when needed."""
        if self.model is None:
            print("Loading Grounding DINO to GPU...")
            self.model = load_model(self.config_path, self.weights_path)

    def unload_model(self):
        """Kills the model and frees GPU memory."""
        if self.model is not None:
            print("Unloading Teacher from GPU to save memory...")
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            gc.collect()

    def label_batch(self, image_paths):
        """Label a batch of images.
        
        If labels_dir was provided at init and matching labels exist,
        copies pre-computed labels instead of running GroundingDINO.
        """
        if self.labels_dir:
            self._label_batch_cached(image_paths)
        else:
            self._label_batch_gdino(image_paths)

    def _label_batch_cached(self, image_paths):
        """Copy pre-existing labels from labels_dir — no GPU needed."""
        print(f"Copying cached labels for {len(image_paths)} images "
              f"(from {self.labels_dir})...")
        
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            name_only = os.path.splitext(filename)[0]
            status = "Cached"
            start_time = time.time()

            try:
                # Copy image
                shutil.copy(img_path, f"{self.output_folder}/images/{filename}")

                # Copy matching label
                src_label = os.path.join(self.labels_dir, f"{name_only}.txt")
                dst_label = f"{self.output_folder}/labels/{name_only}.txt"

                if os.path.exists(src_label):
                    shutil.copy(src_label, dst_label)
                else:
                    # No label found — create empty file (image has no objects)
                    open(dst_label, 'w').close()
                    status = "Cached (no label)"

            except Exception as e:
                print(f"Error copying {filename}: {e}")
                status = f"Error: {e}"

            duration = time.time() - start_time
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([filename, f"{duration:.4f}", status])

    def _label_batch_gdino(self, image_paths):
        """Original GroundingDINO inference path."""
        self._load_model()
        print(f"Teacher is labeling {len(image_paths)} images...")
        
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            status = "Success"
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()

            try:
                image_source, image = load_image(img_path)
                boxes, logits, phrases = predict(
                    model=self.model,
                    image=image,
                    caption=self.text_prompt,
                    box_threshold=0.35,
                    text_threshold=0.25
                )
                
                name_only = os.path.splitext(filename)[0]
                cv2.imwrite(f"{self.output_folder}/images/{filename}", image_source)
                self._save_yolo_file(boxes, phrases, f"{self.output_folder}/labels/{name_only}.txt")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                status = f"Error: {e}"
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            duration = end_time - start_time

            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([filename, f"{duration:.4f}", status])
        
        self.unload_model()

    def _save_yolo_file(self, boxes, phrases, save_path):
        with open(save_path, "w") as f:
            for box, phrase in zip(boxes, phrases):
                class_id = -1
                for key_name, c_id in self.class_map.items():
                    if key_name in phrase:
                        class_id = c_id
                        break
                if class_id != -1:
                    cx, cy, w, h = box.tolist()
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")