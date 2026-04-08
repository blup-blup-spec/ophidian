#!/bin/bash
# ==============================================================================
# Snake Detector — Start Flask Web Server on Raspberry Pi
# TFLite + XNNPACK + INT8 engine (no C++ build needed!)
# Access: http://<PI_IP>:5000
# ==============================================================================

set -e

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  🐍 Snake Detector — TFLite + XNNPACK          ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

BASE_DIR="/home/yoyo/snake_detector"

# ─── Check TFLite runtime is installed ────────────────────────────────────────
python3 -c "
try:
    from tflite_runtime.interpreter import Interpreter
    print('  ✅ tflite-runtime loaded')
except:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
        print('  ✅ tensorflow.lite loaded')
    except:
        print('  ❌ No TFLite runtime! Run: bash scripts/setup_pi_tflite.sh')
        exit(1)
" || exit 1

# ─── Check for .tflite model ─────────────────────────────────────────────────
TFLITE_COUNT=$(find "$BASE_DIR/models" -name "*.tflite" 2>/dev/null | wc -l)

if [ "$TFLITE_COUNT" -eq 0 ]; then
    echo "❌ No .tflite model found in $BASE_DIR/models/"
    echo ""
    echo "   To get the model:"
    echo "   1. Run export_tflite_int8.ipynb in Google Colab"
    echo "   2. Download snake_model_for_pi.zip"
    echo "   3. Upload to Pi:  scp snake_model_for_pi.zip yoyo@<PI_IP>:~/"
    echo "   4. Unzip:  unzip ~/snake_model_for_pi.zip -d $BASE_DIR/models/"
    exit 1
fi

echo "  ✅ Found $TFLITE_COUNT TFLite model(s):"
ls -lh "$BASE_DIR/models/"*.tflite 2>/dev/null | while read line; do
    echo "     $line"
done
echo ""

# ─── Install Python dependencies if needed ────────────────────────────────────
pip3 install --break-system-packages -q flask opencv-python-headless numpy 2>/dev/null || \
pip3 install -q flask opencv-python-headless numpy 2>/dev/null || true

# ─── Set CPU to performance mode ─────────────────────────────────────────────
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1 || true

# ─── Set environment ─────────────────────────────────────────────────────────
export MODEL_DIR="$BASE_DIR/models"

# ─── Start Flask ──────────────────────────────────────────────────────────────
PI_IP=$(hostname -I | awk '{print $1}')
echo "🚀 Starting Flask server..."
echo ""
echo "   ┌──────────────────────────────────────────┐"
echo "   │  Web UI: http://${PI_IP}:5000            │"
echo "   │  Engine: TFLite + XNNPACK + INT8         │"
echo "   │  Press Ctrl+C to stop                    │"
echo "   └──────────────────────────────────────────┘"
echo ""

cd "$BASE_DIR"
python3 pi_server/app.py
