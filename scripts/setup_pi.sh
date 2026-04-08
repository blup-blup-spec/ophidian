#!/bin/bash
# ==============================================================================
# 🐍 Snake Detector — Raspberry Pi 4 Initial Setup
# Run this ONCE on a fresh Raspberry Pi 4 (2GB)
# Optimizes the Pi for maximum inference performance
# ==============================================================================

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  🐍 Raspberry Pi 4 Setup — Snake Detector               ║"
echo "║  Optimizing for edge AI inference                        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ---- Step 1: System Update ----
echo "📦 Step 1/6: Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# ---- Step 2: Install Docker ----
echo ""
echo "🐳 Step 2/6: Installing Docker..."
if command -v docker &> /dev/null; then
    echo "   Docker already installed: $(docker --version)"
else
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker $USER
    echo "   ✅ Docker installed"
fi

# ---- Step 3: Disable Unnecessary Services (save RAM) ----
echo ""
echo "🔧 Step 3/6: Disabling unnecessary services..."
sudo systemctl disable bluetooth 2>/dev/null || true
sudo systemctl stop bluetooth 2>/dev/null || true
sudo systemctl disable avahi-daemon 2>/dev/null || true
sudo systemctl stop avahi-daemon 2>/dev/null || true
sudo systemctl disable triggerhappy 2>/dev/null || true
sudo systemctl stop triggerhappy 2>/dev/null || true
echo "   ✅ Disabled: bluetooth, avahi-daemon, triggerhappy"

# ---- Step 4: Configure GPU Memory (minimize for headless) ----
echo ""
echo "💾 Step 4/6: Setting GPU memory to minimum (16MB)..."
if grep -q "gpu_mem" /boot/config.txt; then
    sudo sed -i 's/gpu_mem=.*/gpu_mem=16/' /boot/config.txt
else
    echo "gpu_mem=16" | sudo tee -a /boot/config.txt > /dev/null
fi
echo "   ✅ GPU memory set to 16MB (frees ~112MB RAM for inference)"

# ---- Step 5: Configure Swap ----
echo ""
echo "💽 Step 5/6: Configuring swap (1GB safety net)..."
sudo dphys-swapfile swapoff 2>/dev/null || true
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=1024/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
echo "   ✅ Swap set to 1024MB"

# ---- Step 6: CPU Performance Mode ----
echo ""
echo "⚡ Step 6/6: Setting CPU governor to performance..."
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null

# Make it persistent across reboots
if ! grep -q "scaling_governor" /etc/rc.local 2>/dev/null; then
    sudo sed -i '/^exit 0$/i echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null' /etc/rc.local 2>/dev/null || true
fi
echo "   ✅ CPU set to performance mode (1.5GHz all cores)"

# ---- Summary ----
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✅ Setup Complete!                                      ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║                                                          ║"
echo "║  Memory optimizations:                                   ║"
echo "║    • GPU memory: 16MB (was 128MB)                        ║"
echo "║    • Bluetooth: disabled                                 ║"
echo "║    • Avahi: disabled                                     ║"
echo "║    • Swap: 1024MB                                        ║"
echo "║                                                          ║"
echo "║  Performance:                                            ║"
echo "║    • CPU governor: performance (1.5GHz)                  ║"
echo "║    • Docker: installed                                   ║"
echo "║                                                          ║"
echo "║  ⚠️  REBOOT REQUIRED for changes to take effect!         ║"
echo "║     Run: sudo reboot                                     ║"
echo "║                                                          ║"
echo "║  After reboot, run:                                      ║"
echo "║     bash scripts/run_on_pi.sh                            ║"
echo "║                                                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
