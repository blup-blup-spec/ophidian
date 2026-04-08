# 🐍 BIO_MIMIC — Venomous Snake Detector (Raspberry Pi 4)

> Real-time venomous vs non-venomous Indian snake detection with bounding boxes  
> YOLOv8n + **TFLite INT8 + XNNPACK** | 15-22 FPS on Pi 4 (2GB RAM)

## Architecture
```
[Google Colab]              → Train YOLOv8n on Kaggle dataset
                                      ↓
                              Export to TFLite INT8
                                      ↓
[Raspberry Pi 4 (2GB)]     → tflite-runtime + XNNPACK → 15-22 FPS detection
                              (only ~5 MB runtime, no PyTorch needed!)
```

## Why TFLite + INT8?

| Metric | Old (NCNN Float32) | New (TFLite INT8) |
|--------|-------------------|-------------------|
| Model size | 12 MB | **~3 MB** |
| Runtime size | ~500 MB (ultralytics) | **~5 MB (tflite-runtime)** |
| FPS on Pi 4 | 5-8 FPS | **15-22 FPS** |
| RAM usage | ~200 MB | **~40 MB** |
| ARM NEON | partial | **full (XNNPACK)** |
| Build needed? | C++ compile on Pi (~10 min) | **No build needed!** |

## Dataset & Training Data
Our model is trained on a highly diverse, custom-curated dataset combining open-source data and proprietary field captures:

- **Kaggle Dataset**: Baseline snake images (`aishanikaggle/indian-snake`) for general feature extraction.
- **Indian Herpetology Database**: High-quality, verified references specifically focused on the **Big Four** venomous species.
- **Custom Field Images (Odisha Rural Terrain)**: Real-world, geographically accurate data collected from rural Odisha to train the model on local backgrounds, camouflage, and typical snake habitats.
- **Infrared (IR) Captures**: Low-light and night-vision frames to ensure the model performs reliably even in dark/nighttime conditions.
- **Annotation Pipeline**: All data was meticulously labeled, unified, and exported in **COCO format** using **Roboflow**, ensuring high precision for YOLO bounding box detection.

### Species Covered (15 Indian species)
| Venomous 🔴 | Non-Venomous 🟢 |
|------------|----------------|
| Spectacled Cobra *(Big Four)* | Python |
| Common Krait *(Big Four)* | Rat Snake |
| Russell's Viper *(Big Four)* | Racer Snake |
| Saw-scaled Viper *(Big Four)* | Keelback |
| Pit Viper | Indian Boa |
| Indian Cat Snake | Kukri |
| | Wolf Snake |
| | Common Trinket |
| | Black Headed Royal Snake |

## Project Structure
```
BIO_MIMIC/
├── colab/
│   ├── train_snake_yolov8.py           ← Training cells for Colab
│   └── export_tflite_int8.ipynb        ← TFLite INT8 export notebook ⚡
├── pi_server/
│   └── app.py                          ← Flask server (TFLite + XNNPACK)
├── inference/                          ← [Legacy] C++ NCNN code
│   ├── CMakeLists.txt
│   ├── snake_detector.h
│   ├── snake_detector.cpp
│   └── main.cpp
├── scripts/
│   ├── setup_pi_tflite.sh              ← One-time Pi setup (installs tflite-runtime)
│   ├── start_flask.sh                  ← Start web server
│   ├── auto_deploy.py                  ← Auto-deploy from PC to Pi
│   └── build_on_pi.sh                  ← [Legacy] C++ build
└── models/                             ← TFLite model files go here
    └── best_int8.tflite                ← INT8 quantized model
```

## Quick Start

### Step 1: Train (Google Colab)
1. Open [Google Colab](https://colab.research.google.com)
2. Copy cells from `colab/train_snake_yolov8.py` into a notebook
3. Run cells 1-7 (training)

### Step 2: Export TFLite INT8 (Google Colab)
1. Open `colab/export_tflite_int8.ipynb` in Colab
2. Run all cells
3. Download `snake_model_for_pi.zip`

### Step 3: Deploy to Pi
```bash
# Option A: Auto-deploy from your PC
python scripts/auto_deploy.py

# Option B: Manual deploy
scp snake_model_for_pi.zip yoyo@<PI_IP>:~/
ssh yoyo@<PI_IP>

# On the Pi:
mkdir -p ~/snake_detector/models
unzip ~/snake_model_for_pi.zip -d ~/snake_detector/models/
bash ~/snake_detector/scripts/setup_pi_tflite.sh   # one-time setup
bash ~/snake_detector/scripts/start_flask.sh        # start server
```

### Step 4: Open Web UI
```
http://<PI_IP>:5000
```

## FPS Display (Fixed)
The web UI shows:
- **Pipeline FPS** — real end-to-end speed (capture → inference → draw)
- **Infer ms** — pure model inference time in milliseconds
- **Model type** — INT8/FP32 indicator

## Tech Stack
| Component | Technology |
|-----------|-----------|
| Training | Google Colab + Ultralytics YOLOv8 |
| Model | YOLOv8n → TFLite INT8 quantized |
| Inference | tflite-runtime + XNNPACK delegate |
| Server | Flask + OpenCV |
| Hardware | Raspberry Pi 4 (2GB RAM) |

## Credits
- Dataset: [aishanikaggle/indian-snake](https://www.kaggle.com/datasets/aishanikaggle/indian-snake)
- Inference: [TensorFlow Lite](https://www.tensorflow.org/lite) + [XNNPACK](https://github.com/google/XNNPACK)
- Model: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
