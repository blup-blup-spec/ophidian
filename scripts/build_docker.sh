#!/bin/bash
# ==============================================================================
# 🐍 Snake Detector — Cross-compile Docker image on your PC
# Run this on your development PC (Windows/Linux/Mac with Docker Desktop)
# This builds an ARM64 Docker image that runs on Raspberry Pi 4
# ==============================================================================

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  🐍 Building Snake Detector for Raspberry Pi 4 (ARM64)  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found! Install Docker Desktop first."
    echo "   https://docs.docker.com/desktop/"
    exit 1
fi

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "📁 Project directory: $PROJECT_DIR"

# Check model files exist
if [ ! -d "models" ] || [ -z "$(ls -A models/ 2>/dev/null | grep -E '\.(param|bin)$')" ]; then
    echo ""
    echo "⚠️  WARNING: No model files found in models/ directory!"
    echo "   You need to:"
    echo "   1. Run the Colab training notebook first"
    echo "   2. Download the NCNN model files (.param + .bin)"
    echo "   3. Put them in the models/ directory"
    echo ""
    echo "   Building anyway (you can mount models at runtime)..."
    echo ""
    
    # Create empty models dir with placeholder
    mkdir -p models
    touch models/.gitkeep
fi

# Setup Docker Buildx for multi-platform builds
echo "🔧 Setting up Docker Buildx..."
docker buildx create --name pibuilder --use 2>/dev/null || docker buildx use pibuilder
docker buildx inspect --bootstrap

# Build for ARM64 (Raspberry Pi 4)
echo ""
echo "🏗️  Building Docker image for linux/arm64..."
echo "   This may take 10-20 minutes on first build..."
echo ""

docker buildx build \
    --platform linux/arm64 \
    -f docker/Dockerfile \
    -t snake-detector:latest \
    --output type=docker,dest=snake-detector-arm64.tar \
    .

echo ""
echo "✅ Build complete!"
echo ""
echo "📦 Output: snake-detector-arm64.tar"
echo "   Size: $(du -h snake-detector-arm64.tar | cut -f1)"
echo ""
echo "📋 Next steps:"
echo "   1. Transfer to Pi:  scp snake-detector-arm64.tar pi@<PI_IP>:~/"
echo "   2. On the Pi, run:  bash scripts/run_on_pi.sh"
echo ""
