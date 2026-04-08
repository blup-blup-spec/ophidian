# Walkthrough: TFLite + XNNPACK + INT8 Migration

## What Changed

Completely replaced the NCNN + C++ + ultralytics stack with a lightweight **TFLite + XNNPACK + INT8** pipeline.

### Files Modified/Created

| File | Action | Purpose |
|------|--------|---------|
| [app.py](file:///c:/IOT%20PROJECTS/BIO_MIMIC/pi_server/app.py) | **Rewritten** | Flask server now uses `tflite-runtime` (~5MB) instead of `ultralytics` (~500MB). Fixed FPS to show real pipeline speed. |
| [export_tflite_int8.ipynb](file:///c:/IOT%20PROJECTS/BIO_MIMIC/colab/export_tflite_int8.ipynb) | **NEW** | Colab notebook: exports INT8 + FP32 TFLite, validates accuracy, packages for Pi |
| [setup_pi_tflite.sh](file:///c:/IOT%20PROJECTS/BIO_MIMIC/scripts/setup_pi_tflite.sh) | **NEW** | Pi setup: installs tflite-runtime, OpenCV, Flask, tunes CPU governor |
| [start_flask.sh](file:///c:/IOT%20PROJECTS/BIO_MIMIC/scripts/start_flask.sh) | **Updated** | No more C++ binary dependency — just Python + TFLite |
| [auto_deploy.py](file:///c:/IOT%20PROJECTS/BIO_MIMIC/scripts/auto_deploy.py) | **Updated** | Deploys TFLite models, runs setup_pi_tflite.sh (no 10-min C++ compile) |
| [README.md](file:///c:/IOT%20PROJECTS/BIO_MIMIC/README.md) | **Updated** | Documents new TFLite architecture |

### FPS Fix
- **Before**: Showed only inference time → misleadingly high FPS numbers
- **After**: Shows **Pipeline FPS** (full capture→infer→draw loop) + inference ms separately

## Deployment Flow

```
[Colab] train_snake_yolov8.py → best.pt
                                    ↓
[Colab] export_tflite_int8.ipynb → best_int8.tflite (3 MB)
                                    ↓
[PC]    auto_deploy.py → uploads to Pi, runs setup_pi_tflite.sh
                                    ↓
[Pi]    start_flask.sh → tflite-runtime + XNNPACK → http://PI:5000
```

## What To Do Next

1. **Open `export_tflite_int8.ipynb` in Google Colab**
2. Run all 5 cells (export INT8 → validate → download)
3. Deploy to Pi: `python scripts/auto_deploy.py`
4. Start server: `bash ~/snake_detector/scripts/start_flask.sh`
