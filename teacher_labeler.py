import os
import torch
import cv2
import gc
import time
import csv
from groundingdino.util.inference import load_model, load_image, predict

class TeacherLabeler:
    def __init__(self, model_config_path, model_weights_path, class_map, output_folder="dataset_active"):
        self.output_folder = output_folder
        self.class_map = class_map
        self.text_prompt = " . ".join(class_map.keys())
        
        # We store paths but DO NOT load the model in __init__ to save RAM
        self.config_path = model_config_path
        self.weights_path = model_weights_path
        self.model = None
        
        os.makedirs(f"{self.output_folder}/images", exist_ok=True)
        os.makedirs(f"{self.output_folder}/labels", exist_ok=True)

        # --- NEW: Initialize CSV Log ---
        self.csv_path = os.path.join(self.output_folder, "annotation_time_log.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                # Header for the CSV
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
        self._load_model() # Load just-in-time
        print(f"Teacher is labeling {len(image_paths)} images...")
        
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            status = "Success"
            
            # --- NEW: Start Timer ---
            # Synchronize GPU to ensure previous ops are done before starting timer
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
                
                # Save Image & Label
                name_only = os.path.splitext(filename)[0]
                cv2.imwrite(f"{self.output_folder}/images/{filename}", image_source)
                
                self._save_yolo_file(boxes, phrases, f"{self.output_folder}/labels/{name_only}.txt")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                status = f"Error: {e}"
            
            # --- NEW: Stop Timer & Log ---
            # Synchronize GPU to ensure inference is actually finished before stopping timer
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            duration = end_time - start_time

            # Write to CSV immediately (safer if script crashes)
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([filename, f"{duration:.4f}", status])
        
        # IMPORTANT: Unload immediately after batch is done
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