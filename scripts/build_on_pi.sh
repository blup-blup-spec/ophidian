#!/bin/bash
# ==============================================================================
# Snake Detector — Native Build Script for Raspberry Pi 4
# Run this DIRECTLY on the Raspberry Pi
# ==============================================================================

set -e

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Snake Detector — Native Build on Pi 4          ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ─── Step 1: Install build tools (NOT ncnn — that comes from source) ─────────
echo "[1/4] Installing build tools..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    make \
    git \
    pkg-config \
    libopencv-dev \
    libgomp1

echo "Done: build tools installed"
echo ""

# ─── Step 2: Build NCNN from source ─────────────────────────────────────────
echo "[2/4] Building NCNN from source (this takes ~10 min on Pi 4)..."
if [ -f "/usr/local/lib/cmake/ncnn/ncnnConfig.cmake" ]; then
    echo "NCNN already installed, skipping..."
else
    cd /tmp
    rm -rf ncnn
    git clone --depth 1 --branch 20240410 https://github.com/Tencent/ncnn.git
    cd ncnn
    git submodule update --init
    mkdir -p build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DNCNN_VULKAN=OFF \
          -DNCNN_BUILD_EXAMPLES=OFF \
          -DNCNN_BUILD_TOOLS=OFF \
          -DNCNN_BUILD_BENCHMARK=OFF \
          -DNCNN_BUILD_TESTS=OFF \
          -DNCNN_OPENMP=ON \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          ..
    make -j4
    sudo make install
    echo "NCNN built and installed"
fi
echo ""

# ─── Step 3: Build the Snake Detector ────────────────────────────────────────
echo "[3/4] Compiling snake_detector..."
cd /home/yoyo/snake_detector

rm -rf build
mkdir build && cd build

cmake -DCMAKE_BUILD_TYPE=Release \
      -Dncnn_DIR=/usr/local/lib/cmake/ncnn \
      -DCMAKE_CXX_FLAGS="-march=armv8-a+crc+crypto -O3 -ffast-math" \
      ../inference

make -j4

echo ""
echo "[4/4] Build complete!"
echo ""
ls -lh /home/yoyo/snake_detector/build/snake_detector
echo ""
echo "======================================="
echo " Ready to run:"
echo "   /home/yoyo/snake_detector/build/snake_detector \\"
echo "     --model-dir /home/yoyo/snake_detector/models \\"
echo "     --headless --camera 0 --size 320 --threads 4"
echo "======================================="
