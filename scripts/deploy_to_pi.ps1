# ==============================================================================
# 🐍 Snake Detector — Deploy to Raspberry Pi from Windows
# Run this on your WINDOWS PC to package and transfer everything to the Pi
# ==============================================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$PiIP,
    
    [string]$PiUser = "pi"
)

$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PackageName = "snake_detector_package"
$ZipPath = "$ProjectDir\$PackageName.zip"

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  🐍 Deploying Snake Detector to Raspberry Pi    ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ─── Step 1: Create package ──────────────────────────────────────────────────
Write-Host "📦 [1/3] Packaging files..." -ForegroundColor Yellow

# Create temp staging folder
$StagingDir = "$ProjectDir\_deploy_staging"
if (Test-Path $StagingDir) { Remove-Item -Recurse -Force $StagingDir }
New-Item -ItemType Directory -Path $StagingDir | Out-Null
New-Item -ItemType Directory -Path "$StagingDir\inference" | Out-Null
New-Item -ItemType Directory -Path "$StagingDir\models" | Out-Null
New-Item -ItemType Directory -Path "$StagingDir\scripts" | Out-Null

# Copy C++ source files
Copy-Item "$ProjectDir\inference\*.cpp"  "$StagingDir\inference\"
Copy-Item "$ProjectDir\inference\*.h"    "$StagingDir\inference\"
Copy-Item "$ProjectDir\inference\CMakeLists.txt" "$StagingDir\inference\"

# Copy model files
Copy-Item "$ProjectDir\models\model.ncnn.*" "$StagingDir\models\"

# Copy build script
Copy-Item "$ProjectDir\scripts\build_on_pi.sh" "$StagingDir\scripts\"

# Create zip
if (Test-Path $ZipPath) { Remove-Item $ZipPath }
Compress-Archive -Path "$StagingDir\*" -DestinationPath $ZipPath -Force

# Cleanup staging
Remove-Item -Recurse -Force $StagingDir

$SizeMB = [math]::Round((Get-Item $ZipPath).Length / 1MB, 1)
Write-Host "   ✅ Package created: $PackageName.zip ($SizeMB MB)" -ForegroundColor Green
Write-Host ""

# ─── Step 2: Transfer to Pi ─────────────────────────────────────────────────
Write-Host "📡 [2/3] Transferring to $PiUser@${PiIP}..." -ForegroundColor Yellow
scp $ZipPath "${PiUser}@${PiIP}:~/"

if ($LASTEXITCODE -ne 0) {
    Write-Host "   ❌ SCP failed! Make sure:" -ForegroundColor Red
    Write-Host "      - Pi is on and connected to the same network" -ForegroundColor Red
    Write-Host "      - SSH is enabled on the Pi" -ForegroundColor Red
    Write-Host "      - IP address $PiIP is correct" -ForegroundColor Red
    exit 1
}
Write-Host "   ✅ Transfer complete!" -ForegroundColor Green
Write-Host ""

# ─── Step 3: Print instructions for Pi ───────────────────────────────────────
Write-Host "🚀 [3/3] Now SSH into your Pi and run these commands:" -ForegroundColor Yellow
Write-Host ""
Write-Host "   ssh ${PiUser}@${PiIP}" -ForegroundColor White
Write-Host ""
Write-Host "   Then on the Pi, run:" -ForegroundColor Yellow
Write-Host "   ─────────────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host '   cd ~' -ForegroundColor White
Write-Host '   sudo apt-get install -y unzip' -ForegroundColor White
Write-Host '   unzip -o snake_detector_package.zip -d ~/snake_detector' -ForegroundColor White
Write-Host '   chmod +x ~/snake_detector/scripts/build_on_pi.sh' -ForegroundColor White
Write-Host '   bash ~/snake_detector/scripts/build_on_pi.sh' -ForegroundColor White
Write-Host "   ─────────────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host ""
Write-Host "   After the build (~8-12 min), run the detector:" -ForegroundColor Yellow
Write-Host "   ─────────────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host '   ~/snake_detector/build/snake_detector \' -ForegroundColor White
Write-Host '     --model-dir ~/snake_detector/models \' -ForegroundColor White
Write-Host '     --headless --camera 0 --size 320 --threads 4' -ForegroundColor White
Write-Host "   ─────────────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host ""

# Cleanup zip
# Remove-Item $ZipPath  # Uncomment to auto-delete after transfer

Write-Host "✅ Deployment complete!" -ForegroundColor Green
Write-Host ""
