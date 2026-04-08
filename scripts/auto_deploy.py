import os
import shutil
import zipfile
import paramiko
from scp import SCPClient
import sys
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

PI_IP = os.getenv("PI_IP", "10.230.19.146")
PI_USER = os.getenv("PI_USER", "yoyo")
PI_PASS = os.getenv("PI_PASS")
PI_PORT = int(os.getenv("PI_PORT", "2222"))

# Validate required credentials
if not PI_PASS:
    print("❌ Error: PI_PASS not found in environment or .env file")
    print("   Please create a .env file with your Pi credentials (see .env.example)")
    sys.exit(1)

print("=" * 55)
print(" 🐍 SNAKE DETECTOR — AUTO DEPLOY (TFLite + XNNPACK)")
print("=" * 55)

# ─── Step 1: Package ─────────────────────────────────────────────────────────
print("\n[1/3] Packaging files...")
staging = "_deploy_staging"
if os.path.exists(staging):
    shutil.rmtree(staging)
os.makedirs(os.path.join(staging, "models"))
os.makedirs(os.path.join(staging, "scripts"))
os.makedirs(os.path.join(staging, "pi_server"))

# Copy TFLite models (primary) + NCNN models (legacy fallback)
model_count = 0
for f in os.listdir("models"):
    if f.endswith('.tflite') or f.startswith("model.ncnn."):
        shutil.copy(os.path.join("models", f), os.path.join(staging, "models"))
        model_count += 1
        print(f"  Model: {f}")

# Also check snake_model_for_pi for tflite files
if os.path.isdir("snake_model_for_pi"):
    for f in os.listdir("snake_model_for_pi"):
        if f.endswith('.tflite'):
            dst = os.path.join(staging, "models", f)
            if not os.path.exists(dst):
                shutil.copy(os.path.join("snake_model_for_pi", f), dst)
                model_count += 1
                print(f"  Model: {f} (from snake_model_for_pi)")

if model_count == 0:
    print("  ⚠️  No model files found! Deploy will need models uploaded separately.")

# Copy scripts
for script in ["setup_pi_tflite.sh", "start_flask.sh", "build_on_pi.sh"]:
    src = os.path.join("scripts", script)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(staging, "scripts"))

# Copy Flask server
for f in os.listdir("pi_server"):
    shutil.copy(os.path.join("pi_server", f), os.path.join(staging, "pi_server"))

# Create zip
zipname = "snake_detector_package.zip"
with zipfile.ZipFile(zipname, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(staging):
        for file in files:
            file_path = os.path.join(root, file)
            zip_path = os.path.relpath(file_path, staging)
            zipf.write(file_path, zip_path)

shutil.rmtree(staging)
size_mb = os.path.getsize(zipname) / (1024 * 1024)
print(f"\n   Package: {zipname} ({size_mb:.1f} MB)")

# ─── Step 2: Connect & Upload ────────────────────────────────────────────────
print(f"\n[2/3] Connecting to {PI_USER}@{PI_IP}:{PI_PORT}...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    ssh.connect(PI_IP, username=PI_USER, password=PI_PASS, port=PI_PORT, timeout=15)
    print("   Connected!")

    print("   Uploading package...")
    with SCPClient(ssh.get_transport()) as scp_client:
        scp_client.put(zipname, remote_path=f"/home/{PI_USER}/")
    print("   Upload done!")

    # ─── Step 3: Unzip and Setup TFLite ───────────────────────────────────────
    print(f"\n[3/3] Deploying on Pi...")
    print("-" * 50)

    commands = f"""
cd /home/{PI_USER}
rm -rf /home/{PI_USER}/snake_detector
unzip -o /home/{PI_USER}/snake_detector_package.zip -d /home/{PI_USER}/snake_detector
chmod +x /home/{PI_USER}/snake_detector/scripts/*.sh
sed -i 's/\\r$//' /home/{PI_USER}/snake_detector/scripts/*.sh
cd /home/{PI_USER}/snake_detector
bash /home/{PI_USER}/snake_detector/scripts/setup_pi_tflite.sh
"""

    stdin, stdout, stderr = ssh.exec_command(commands, get_pty=True, timeout=600)
    stdin.write(PI_PASS + "\n")
    stdin.flush()

    for line in iter(stdout.readline, ""):
        print(line, end="")
        sys.stdout.flush()

    exit_status = stdout.channel.recv_exit_status()
    if exit_status == 0:
        print("\n" + "=" * 55)
        print(" ✅ DEPLOY SUCCESSFUL! (TFLite + XNNPACK)")
        print("=" * 55)
        print("\n Now start the Flask server:")
        print(f"   ssh -p {PI_PORT} {PI_USER}@{PI_IP}")
        print(f"   bash ~/snake_detector/scripts/start_flask.sh")
        print(f"\n Then open: http://{PI_IP}:5000")
    else:
        print(f"\n ❌ DEPLOY FAILED (exit code {exit_status})")
        err = stderr.read().decode()
        if err:
            print("STDERR:", err)

except Exception as e:
    print(f"\n ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
finally:
    ssh.close()
    print("\nSSH connection closed.")
