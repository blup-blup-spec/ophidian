#!/bin/bash
# ==============================================================================
# 🐍 Snake Detector — Run on Raspberry Pi 4
# Execute this script on the Raspberry Pi after transferring the Docker image
# ==============================================================================

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  🐍 Snake Detector — Launching on Raspberry Pi 4        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found! Installing Docker..."
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker $USER
    echo ""
    echo "✅ Docker installed. Please log out and log back in, then run this script again."
    exit 0
fi

# Load the Docker image if tar file exists
if [ -f "$HOME/snake-detector-arm64.tar" ]; then
    echo "📦 Loading Docker image from tar file..."
    docker load -i $HOME/snake-detector-arm64.tar
    echo "✅ Image loaded successfully"
    echo ""
fi

# Check camera
echo "📷 Checking camera..."
if [ -e /dev/video0 ]; then
    echo "   ✅ Camera found at /dev/video0"
else
    echo "   ⚠️  No camera at /dev/video0"
    echo "   Trying to detect cameras..."
    ls -la /dev/video* 2>/dev/null || echo "   ❌ No video devices found!"
    echo "   Make sure your camera is connected."
fi

# Set CPU governor to performance
echo "⚡ Setting CPU to performance mode..."
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1 || true

# Run the detector
echo ""
echo "🚀 Starting Snake Detector..."
echo "   Mode: Headless (no display) for maximum FPS"
echo "   Press Ctrl+C to stop"
echo ""

docker run -it --rm \
    --name snake-detector \
    --device=/dev/video0:/dev/video0 \
    --memory=1500m \
    --cpus=4 \
    -e OMP_NUM_THREADS=4 \
    -e NCNN_THREADS=4 \
    snake-detector:latest \
    --model-dir /app/models \
    --headless \
    --camera 0 \
    --size 320 \
    --conf 0.45 \
    --threads 4
