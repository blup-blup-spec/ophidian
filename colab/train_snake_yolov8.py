"""
================================================================================
 🐍 SNAKE DETECTOR — COMPLETE TRAINING NOTEBOOK
 
 Dataset: aishanikaggle/indian-snake (Kaggle)
 Output:  Bounding boxes + venomous/non_venomous labels (like your screenshot)
 Deploy:  Raspberry Pi 4 (2GB) via NCNN
 
 HOW TO USE:
 1. Open Google Colab: https://colab.research.google.com
 2. Go to Runtime → Change runtime type → T4 GPU
 3. Create a new notebook
 4. Copy EACH CELL below into a separate Colab cell
 5. Run them in order (Cell 1, Cell 2, Cell 3...)
 
 Each cell starts with: # ====== CELL X ======
 Each cell ends with:   # ====== END CELL X ======
================================================================================
"""


# ====== CELL 1: Install dependencies ======
# Copy everything between the lines into one Colab cell

"""
!pip install -q ultralytics==8.3.40 opendatasets supervision
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU - fix this first!'}")
"""

# ====== END CELL 1 ======


# ====== CELL 2: Download the Kaggle dataset ======
# It will ask for your Kaggle username and API key
# Get your key from: kaggle.com → Your Profile → Settings → API → Create New Token

"""
import opendatasets as od
od.download("https://www.kaggle.com/datasets/aishanikaggle/indian-snake")
"""

# ====== END CELL 2 ======


# ====== CELL 3: Auto-generate bounding boxes + organize for YOLO ======
# The Kaggle dataset has images in folders but NO bounding boxes.
# YOLO detection needs bounding boxes. This cell creates them automatically
# by using a pretrained YOLOv8 model to find where the snake is in each image.

"""
import os, shutil, glob, random, yaml
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

# ── Species → venomous/non_venomous mapping ──
SPECIES_MAP = {
    "COBRA_VENOUMOUS":                        "venomous",
    "COMMON_KRAIT-VENOMOUS":                  "venomous",
    "RUSSELL_VIPPER_VENOMOUS":                "venomous",
    "SAW_SCALED_VIPPER_VENOMOUS":             "venomous",
    "PIT_VIPPER_VENOMOUS":                    "venomous",
    "INDIAN_CAT_VEN":                         "venomous",
    "BLACK_HEADED_ROYAL_SNAKE_NON-VENOUMOUS": "non_venomous",
    "COMMON_TRINKET_NON-VENOUMOUS":           "non_venomous",
    "INDIAN_BOA_NON-VENOUMOUS":               "non_venomous",
    "KEELBACK_NON-VENOMOUS":                  "non_venomous",
    "KUKRI_NON-VENOMOUS":                     "non_venomous",
    "PYTHON_NON-VENOMOUS":                    "non_venomous",
    "RACER_SNAKE_NON-VENOMOUS":               "non_venomous",
    "RAT_SNAKE_NON-VENOMOUS":                 "non_venomous",
    "WOLF_SNAKE_NON-VENOMOUS":                "non_venomous",
}
TRAIN_MAP = {
    "cobra":"venomous","krait":"venomous","russell":"venomous",
    "saw":"venomous","pit":"venomous","cat":"venomous",
    "black":"non_venomous","common":"non_venomous","boa":"non_venomous",
    "keelback":"non_venomous","kukri":"non_venomous","python":"non_venomous",
    "racer":"non_venomous","rat":"non_venomous","wolf":"non_venomous",
}

CLASS_TO_ID = {"venomous": 0, "non_venomous": 1}
IMAGE_EXT = {'.jpg','.jpeg','.png','.bmp','.webp'}
SRC = "indian-snake"
OUT = "snake_yolo_dataset"
SPLIT = 0.80

# ── Step 1: Collect all images with their class ──
all_images = []  # list of (path, class_name)

# From root species folders (most data)
for folder_name, cls in SPECIES_MAP.items():
    folder = os.path.join(SRC, folder_name)
    if not os.path.isdir(folder):
        # fuzzy match
        for d in os.listdir(SRC):
            if d.upper().startswith(folder_name.upper()[:12]) and os.path.isdir(os.path.join(SRC, d)):
                folder = os.path.join(SRC, d)
                break
    if not os.path.isdir(folder):
        print(f"  Skip: {folder_name}")
        continue
    for f in os.listdir(folder):
        if Path(f).suffix.lower() in IMAGE_EXT:
            all_images.append((os.path.join(folder, f), cls))

# From train/valid subfolders
seen = set(os.path.basename(p).lower() for p, _ in all_images)
for sub in ["train", "valid"]:
    sub_path = os.path.join(SRC, sub)
    if not os.path.isdir(sub_path):
        continue
    for species_dir in os.listdir(sub_path):
        sp = os.path.join(sub_path, species_dir)
        if not os.path.isdir(sp):
            continue
        cls = TRAIN_MAP.get(species_dir.lower())
        if cls is None:
            for k, v in TRAIN_MAP.items():
                if k in species_dir.lower():
                    cls = v
                    break
        if cls is None:
            continue
        for f in os.listdir(sp):
            if Path(f).suffix.lower() in IMAGE_EXT and f.lower() not in seen:
                all_images.append((os.path.join(sp, f), cls))
                seen.add(f.lower())

ven = sum(1 for _, c in all_images if c == "venomous")
nven = sum(1 for _, c in all_images if c == "non_venomous")
print(f"Collected: {len(all_images)} images ({ven} venomous, {nven} non_venomous)")

# ── Step 2: Auto-generate bounding boxes ──
# Use pretrained YOLOv8n (trained on COCO) to detect objects.
# Since most snake photos have the snake as the main subject,
# we use a fallback: if no animal is detected, use center 80% of image as the box.

print("\\nGenerating bounding boxes (this takes a few minutes)...")
auto_model = YOLO("yolov8n.pt")  # pretrained on COCO (80 classes)
# COCO animal classes that could match snakes/reptiles
ANIMAL_IDS = {14,15,16,17,18,19,20,21,22,23,24,25}  # bird,cat,dog,horse,...

annotations = {}  # path -> (cx, cy, w, h) normalized YOLO format
fallback_count = 0
detected_count = 0

for idx, (img_path, cls) in enumerate(all_images):
    if idx % 200 == 0:
        print(f"  Processing {idx+1}/{len(all_images)}...")
    
    try:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Try auto-detection
        results = auto_model.predict(img_path, imgsz=320, conf=0.2, verbose=False)
        
        best_box = None
        best_area = 0
        
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    # Take the largest detected object (likely the snake)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    area = (x2-x1) * (y2-y1)
                    if area > best_area:
                        best_area = area
                        best_box = (x1, y1, x2, y2)

        if best_box is not None and best_area > (w * h * 0.01):
            # Use detected box
            x1, y1, x2, y2 = best_box
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            # Clamp
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            bw = max(0.05, min(1, bw))
            bh = max(0.05, min(1, bh))
            detected_count += 1
        else:
            # Fallback: snake photos usually have snake in center ~80% of frame
            cx, cy, bw, bh = 0.5, 0.5, 0.85, 0.85
            fallback_count += 1
        
        annotations[img_path] = (cls, cx, cy, bw, bh)
    except:
        # If anything fails, use center crop
        annotations[img_path] = (cls, 0.5, 0.5, 0.85, 0.85)
        fallback_count += 1

print(f"\\nBounding boxes: {detected_count} auto-detected, {fallback_count} center-fallback")

# ── Step 3: Create YOLO dataset structure ──
for split in ["train", "val"]:
    os.makedirs(f"{OUT}/{split}/images", exist_ok=True)
    os.makedirs(f"{OUT}/{split}/labels", exist_ok=True)

# Shuffle and split
items = list(annotations.items())
random.seed(42)
random.shuffle(items)
split_idx = int(len(items) * SPLIT)
train_items = items[:split_idx]
val_items = items[split_idx:]

def write_split(items, split_name):
    for i, (img_path, (cls, cx, cy, bw, bh)) in enumerate(items):
        ext = Path(img_path).suffix
        fname = f"{split_name}_{i:05d}"
        
        # Copy image
        dst_img = f"{OUT}/{split_name}/images/{fname}{ext}"
        shutil.copy2(img_path, dst_img)
        
        # Write YOLO label (class_id cx cy w h)
        class_id = CLASS_TO_ID[cls]
        dst_lbl = f"{OUT}/{split_name}/labels/{fname}.txt"
        with open(dst_lbl, "w") as f:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\\n")

write_split(train_items, "train")
write_split(val_items, "val")

# ── Step 4: Create data.yaml ──
data_yaml = {
    "path": os.path.abspath(OUT),
    "train": "train/images",
    "val": "val/images",
    "nc": 2,
    "names": ["venomous", "non_venomous"]
}

yaml_path = f"{OUT}/data.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"\\n✅ Dataset ready: {OUT}/")
print(f"   Train: {len(train_items)} images")
print(f"   Val:   {len(val_items)} images")
print(f"   data.yaml: {yaml_path}")
print(f"   Classes: venomous (0), non_venomous (1)")
"""

# ====== END CELL 3 ======


# ====== CELL 4: Train YOLOv8n detection ======

"""
from ultralytics import YOLO
import os

HOME = os.getcwd()

model = YOLO("yolov8n.pt")

results = model.train(
    data=f"{HOME}/snake_yolo_dataset/data.yaml",
    imgsz=320,
    epochs=150,
    batch=16,
    device=0,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    augment=True,
    mosaic=1.0,
    mixup=0.2,
    degrees=30.0,
    translate=0.2,
    scale=0.5,
    flipud=0.5,
    fliplr=0.5,
    hsv_h=0.02,
    hsv_s=0.7,
    hsv_v=0.4,
    patience=30,
    cos_lr=True,
    project="runs",
    name="snake_det",
    exist_ok=True,
    verbose=True,
)

print(f"\\n✅ TRAINING DONE!")
print(f"   Best weights: {HOME}/runs/snake_det/weights/best.pt")
"""

# ====== END CELL 4 ======


# ====== CELL 5: Validate ======

"""
model = YOLO(f"{HOME}/runs/snake_det/weights/best.pt")
metrics = model.val(data=f"{HOME}/snake_yolo_dataset/data.yaml", imgsz=320)
print(f"\\nmAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
"""

# ====== END CELL 5 ======


# ====== CELL 6: Test — see bounding boxes on images ======

"""
import glob
from IPython.display import Image, display

# Predict on validation images
results = model.predict(
    source=f"{HOME}/snake_yolo_dataset/val/images",
    imgsz=320,
    conf=0.4,
    save=True,
    project="runs",
    name="predictions",
    exist_ok=True,
)

# Show results with BOUNDING BOXES
pred_imgs = sorted(glob.glob(f"{HOME}/runs/predictions/*.jpg"))[:8]
for p in pred_imgs:
    print(os.path.basename(p))
    display(Image(filename=p, width=500))
    print()

print("👆 You should see boxes around snakes with 'venomous 0.XX' or 'non_venomous 0.XX'")
"""

# ====== END CELL 6 ======


# ====== CELL 7: Test on a video (optional) ======

"""
# Upload a snake video to test real-time detection
from google.colab import files
print("Upload a snake video (.mp4/.avi):")
try:
    uploaded = files.upload()
    video = list(uploaded.keys())[0]
    results = model.predict(source=video, imgsz=320, conf=0.4, save=True,
                            project="runs", name="video_test", exist_ok=True)
    print(f"\\n✅ Video result saved in runs/video_test/")
    print("Download it to check the bounding boxes on video!")
except:
    print("No video uploaded — skipping. You can test on Pi instead.")
"""

# ====== END CELL 7 ======


# ====== CELL 8: Export to NCNN (for Raspberry Pi) ======

"""
print("Exporting to NCNN format...")
model = YOLO(f"{HOME}/runs/snake_det/weights/best.pt")

ncnn_path = model.export(format="ncnn", imgsz=320, half=False)
print(f"\\n✅ NCNN exported: {ncnn_path}")

# Also export ONNX as backup
onnx_path = model.export(format="onnx", imgsz=320)
print(f"✅ ONNX exported: {onnx_path}")
"""

# ====== END CELL 8 ======


# ====== CELL 9: Check NCNN files ======

"""
import os
ncnn_dir = None
for root, dirs, files_list in os.walk("runs"):
    for f in files_list:
        if f.endswith(".ncnn.param"):
            ncnn_dir = root
            break
    if ncnn_dir: break

print(f"NCNN model: {ncnn_dir}")
for f in sorted(os.listdir(ncnn_dir)):
    sz = os.path.getsize(os.path.join(ncnn_dir, f)) / 1024 / 1024
    print(f"  {f} ({sz:.2f} MB)")
"""

# ====== END CELL 9 ======


# ====== CELL 10: Download for Raspberry Pi ======

"""
import shutil

deploy = "snake_model_for_pi"
if os.path.exists(deploy): shutil.rmtree(deploy)
os.makedirs(deploy)

# Copy NCNN model
for f in os.listdir(ncnn_dir):
    shutil.copy(os.path.join(ncnn_dir, f), deploy)

# Copy best.pt too
shutil.copy(f"{HOME}/runs/snake_det/weights/best.pt", deploy)

# Model info
with open(f"{deploy}/README.txt", "w") as fp:
    fp.write("Snake Detector Model\\n")
    fp.write("Model: YOLOv8n (Detection with bounding boxes)\\n")
    fp.write("Input: 320x320\\n")
    fp.write("Classes: 0=venomous, 1=non_venomous\\n")
    fp.write(f"mAP50: {metrics.box.map50:.4f}\\n")
    fp.write("Dataset: aishanikaggle/indian-snake\\n")

shutil.make_archive(deploy, "zip", deploy)

from google.colab import files
files.download(f"{deploy}.zip")
print("\\n✅ Downloading snake_model_for_pi.zip")
print("Put .param + .bin files in BIO_MIMIC/models/")
"""

# ====== END CELL 10 ======
