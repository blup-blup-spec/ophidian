"""
Deploy & Start — Snake Detector on Raspberry Pi
1. Upload new Flask app
2. Install ultralytics (uses NCNN internally)
3. Kill old server, start new one
"""
import paramiko
from scp import SCPClient
import sys
import time
import os

PI = {
    "host": os.getenv("PI_IP", "10.230.19.146"),
    "user": os.getenv("PI_USER", "yoyo"),
    "pass": os.getenv("PI_PASS"),
    "port": int(os.getenv("PI_PORT", 2222))
}

if not PI["pass"]:
    print("❌ Error: PI_PASS not found in environment (consider setting it or using a .env file)")
    sys.exit(1)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(PI["host"], username=PI["user"], password=PI["pass"], port=PI["port"], timeout=15)
print("=== Connected to Pi ===\n")

# 1. Upload
print("[1/4] Uploading app.py...")
with SCPClient(ssh.get_transport()) as scp:
    scp.put("pi_server/app.py", "/home/yoyo/snake_detector/pi_server/app.py")
ssh.exec_command("sed -i 's/\\r$//' /home/yoyo/snake_detector/pi_server/app.py")
time.sleep(1)
print("  Done\n")

# 2. Install ultralytics
print("[2/4] Installing ultralytics (this may take 2-3 min)...")
cmd = "pip3 install --break-system-packages ultralytics 2>&1 | tail -8"
stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True, timeout=600)
for line in iter(stdout.readline, ""):
    print("  " + line, end="")
    sys.stdout.flush()
exit_code = stdout.channel.recv_exit_status()
print(f"  Install exit: {exit_code}\n")

# 3. Kill old, start new
print("[3/4] Restarting Flask...")
ssh.exec_command("pkill -f 'python3 pi_server/app.py'; pkill rpicam-vid; sleep 2")
time.sleep(3)

start_cmd = (
    "cd /home/yoyo/snake_detector && "
    "MODEL_DIR=/home/yoyo/snake_detector/models "
    "nohup python3 pi_server/app.py > /tmp/flask_snake.log 2>&1 & disown"
)
transport = ssh.get_transport()
ch = transport.open_session()
ch.exec_command(start_cmd)

# 4. Wait for Flask to bind
print("[4/4] Waiting for Flask to start (model loading takes ~10-15s)...")
for i in range(20):
    time.sleep(5)
    stdin, stdout, stderr = ssh.exec_command("ss -tlnp | grep 5000")
    port = stdout.read().decode().strip()
    if "5000" in port:
        print(f"  ✅ Flask is LIVE on :5000")
        break
    # Show log progress
    stdin2, stdout2, stderr2 = ssh.exec_command("tail -3 /tmp/flask_snake.log 2>/dev/null")
    log = stdout2.read().decode().strip()
    print(f"  [{(i+1)*5}s] {log[-80:] if log else 'starting...'}")

# Show final log
stdin3, stdout3, stderr3 = ssh.exec_command("cat /tmp/flask_snake.log 2>/dev/null | head -25")
print(f"\n--- Flask Log ---\n{stdout3.read().decode()}")

ssh.close()
print(f"\n{'='*50}")
print(f"  Open: http://{PI['host']}:5000")
print(f"  Detection FPS shown is REAL (not camera FPS)")
print(f"{'='*50}")
