#!/bin/bash
# ==============================================================================
# Snake Detector — Pi 4 Setup for TFLite + XNNPACK
# Run this ONCE on the Raspberry Pi before starting the Flask server
# ==============================================================================

set -e

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  🐍 Snake Detector — TFLite Setup for Pi 4     ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

BASE_DIR="/home/yoyo/snake_detector"

# ─── Step 1: System packages ─────────────────────────────────────────────────
echo "[1/4] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-numpy \
    libopencv-dev \
    python3-opencv \
    libatlas-base-dev \
    libhdf5-dev

echo "✅ System packages installed"
echo ""

# ─── Step 2: Install tflite-runtime (tiny! ~5 MB vs ~500 MB for ultralytics) ─
echo "[2/4] Installing tflite-runtime..."

# Try standard pip first, then ARM-specific index
pip3 install --break-system-packages tflite-runtime 2>/dev/null || \
pip3 install tflite-runtime 2>/dev/null || \
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime 2>/dev/null || \
{
    echo "  Trying ARM64 wheel directly..."
    PYVER=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
    pip3 install --break-system-packages \
        "https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-${PYVER}-${PYVER}-linux_aarch64.whl" 2>/dev/null || \
    pip3 install tflite_runtime
}

echo "✅ tflite-runtime installed"
echo ""

# ─── Step 3: Install Flask + dependencies ────────────────────────────────────
echo "[3/4] Installing Flask + dependencies..."
pip3 install --break-system-packages flask opencv-python-headless numpy 2>/dev/null || \
pip3 install flask opencv-python-headless numpy

echo "✅ Flask installed"
echo ""

# ─── Step 4: Verify installation ─────────────────────────────────────────────
echo "[4/4] Verifying setup..."
echo ""

python3 -c "
import sys
print(f'Python: {sys.version}')

try:
    from tflite_runtime.interpreter import Interpreter
    print('✅ tflite-runtime: OK')
except:
    try:
        import tensorflow as tf
        print(f'✅ tensorflow: {tf.__version__}')
    except:
        print('❌ No TFLite runtime found!')
        sys.exit(1)

import cv2
print(f'✅ OpenCV: {cv2.__version__}')

import numpy as np
print(f'✅ NumPy: {np.__version__}')

import flask
print(f'✅ Flask: {flask.__version__}')

print()
print('All dependencies OK!')
"

# ─── Check for model files ───────────────────────────────────────────────────
echo ""
echo "Checking models..."
TFLITE_COUNT=$(find "$BASE_DIR/models" -name "*.tflite" 2>/dev/null | wc -l)

if [ "$TFLITE_COUNT" -gt 0 ]; then
    echo "✅ Found $TFLITE_COUNT TFLite model(s):"
    ls -lh "$BASE_DIR/models/"*.tflite 2>/dev/null
else
    echo "⚠️  No .tflite models found in $BASE_DIR/models/"
    echo "    Upload your model: scp snake_model_for_pi.zip yoyo@<PI_IP>:~/"
    echo "    Then: unzip ~/snake_model_for_pi.zip -d $BASE_DIR/models/"
fi

# ─── Performance tuning ─────────────────────────────────────────────────────
echo ""
echo "Applying Pi 4 performance tuning..."

# Set CPU governor to performance (if available)
if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1 || true
    echo "  CPU governor: performance"
fi

# Increase GPU memory split for camera
if grep -q "gpu_mem" /boot/config.txt 2>/dev/null; then
    echo "  GPU memory: already configured"
else
    echo "  Note: Add 'gpu_mem=128' to /boot/config.txt for camera support"
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo " ✅ SETUP COMPLETE!"
echo ""
echo " Runtime: tflite-runtime (~5 MB, not ~500 MB)"
echo " Delegate: XNNPACK (ARM NEON acceleration)"
echo ""
echo " Start the server:"
echo "   bash $BASE_DIR/scripts/start_flask.sh"
echo "═══════════════════════════════════════════════════"
