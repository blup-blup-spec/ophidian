#!/usr/bin/env python3
"""
Snake Detector — TFLite + XNNPACK + INT8
=========================================
Lightweight real-time inference on Raspberry Pi 4 (2GB RAM)

Runtime: tflite-runtime (~5 MB) instead of ultralytics + PyTorch (~500 MB)
Delegate: XNNPACK for ARM NEON acceleration (float models)
Quantization: Full INT8 for maximum speed on ARM Cortex-A72

Access: http://<PI_IP>:5000
"""

import time
import os
import sys
import threading
import base64
from flask import Flask, Response, render_template_string, jsonify, request

import cv2
import numpy as np

# ─── TFLite Runtime (lightweight, no PyTorch/TensorFlow needed) ──────────────
try:
    from tflite_runtime.interpreter import Interpreter
    from tflite_runtime.interpreter import load_delegate
    TFLITE_SOURCE = "tflite-runtime"
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
        from tensorflow.lite.python.interpreter import load_delegate
        TFLITE_SOURCE = "tensorflow"
    except ImportError:
        print("FATAL: Install tflite-runtime:  pip3 install tflite-runtime")
        sys.exit(1)

app = Flask(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────
MODEL_DIR = os.environ.get("MODEL_DIR", r"c:\IOT PROJECTS\BIO_MIMIC\snake_model_for_pi (3)")
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 30
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
IMG_SIZE = 320
NUM_THREADS = 4
CLASS_NAMES = ["venomous", "non_venomous"]

# ─── Global State ────────────────────────────────────────────────────────────
latest_frame = None
latest_detections = []
latest_pipeline_fps = 0.0
latest_inference_ms = 0.0
frame_lock = threading.Lock()
running = True

# TFLite interpreter
interp = None
input_details = None
output_details = None
model_dtype = "float32"
model_name = ""


# =============================================================================
#  MODEL LOADING
# =============================================================================

def find_tflite_model():
    """Find .tflite model file in MODEL_DIR"""
    if not os.path.isdir(MODEL_DIR):
        print(f"[-] Model directory not found: {MODEL_DIR}")
        return None

    # Priority order: int8 > float16 > float32
    priority = ["int8", "quant", "full_integer", "float16", "fp16"]
    tflite_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.tflite')]

    if not tflite_files:
        print(f"[-] No .tflite files in {MODEL_DIR}")
        print("    Export from Colab:  model.export(format='tflite', int8=True)")
        return None

    # Try to find INT8 model first
    for keyword in priority:
        for f in tflite_files:
            if keyword in f.lower():
                return os.path.join(MODEL_DIR, f)

    # Fall back to first .tflite file
    return os.path.join(MODEL_DIR, tflite_files[0])


def load_model():
    """Load TFLite model with XNNPACK delegate for ARM NEON acceleration"""
    global interp, input_details, output_details, model_dtype, model_name

    model_path = find_tflite_model()
    if model_path is None:
        sys.exit(1)

    model_name = os.path.basename(model_path)
    print(f"[+] Loading TFLite model: {model_path}")
    print(f"    Runtime: {TFLITE_SOURCE}")

    # Try XNNPACK delegate (accelerates float ops via ARM NEON)
    xnnpack_loaded = False
    try:
        interp = Interpreter(
            model_path=model_path,
            num_threads=NUM_THREADS,
            experimental_delegates=[load_delegate('libXNNPACK.so')]
        )
        xnnpack_loaded = True
        print("[+] XNNPACK delegate loaded — ARM NEON acceleration ON")
    except Exception:
        try:
            interp = Interpreter(
                model_path=model_path,
                num_threads=NUM_THREADS,
                experimental_delegates=[load_delegate('XNNPACK')]
            )
            xnnpack_loaded = True
            print("[+] XNNPACK delegate loaded — ARM NEON acceleration ON")
        except Exception:
            # XNNPACK may be compiled into tflite-runtime already
            interp = Interpreter(
                model_path=model_path,
                num_threads=NUM_THREADS
            )
            print("[+] Default TFLite runtime (XNNPACK may be built-in)")

    interp.allocate_tensors()
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    inp = input_details[0]
    out = output_details[0]

    print(f"[+] Input:  shape={inp['shape']}  dtype={inp['dtype'].__name__}")
    print(f"[+] Output: shape={out['shape']}  dtype={out['dtype'].__name__}")
    print(f"[+] Threads: {NUM_THREADS}")

    if inp['dtype'] == np.uint8 or inp['dtype'] == np.int8:
        model_dtype = "INT8"
        print("[+] ⚡ INT8 quantized model — maximum ARM performance!")
    elif inp['dtype'] == np.float16:
        model_dtype = "FP16"
        print("[+] Float16 model")
    else:
        model_dtype = "FP32"
        if xnnpack_loaded:
            print("[+] Float32 model — XNNPACK accelerating via NEON")
        else:
            print("[+] Float32 model")

    # Warm up (first inference is always slow due to memory allocation)
    dummy = np.zeros(inp['shape'], dtype=inp['dtype'])
    interp.set_tensor(inp['index'], dummy)
    interp.invoke()

    # Run 3 warm-up inferences for stable timing
    for _ in range(3):
        interp.set_tensor(inp['index'], dummy)
        interp.invoke()

    print("[+] Model warmed up — ready for inference\n")


# =============================================================================
#  PREPROCESSING
# =============================================================================

def preprocess(frame):
    """
    Letterbox resize + normalize for YOLOv8 TFLite input.
    Returns: (input_tensor, scale, pad_w, pad_h, orig_w, orig_h)
    """
    h, w = frame.shape[:2]
    scale = min(IMG_SIZE / w, IMG_SIZE / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to square with gray (114)
    pad_w = (IMG_SIZE - new_w) // 2
    pad_h = (IMG_SIZE - new_h) // 2
    padded = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    # BGR → RGB (YOLOv8 expects RGB)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    # Format based on model input dtype
    inp_dtype = input_details[0]['dtype']
    if inp_dtype == np.float32:
        blob = rgb.astype(np.float32) / 255.0
    elif inp_dtype == np.uint8:
        blob = rgb  # already uint8 [0, 255]
    elif inp_dtype == np.int8:
        # Apply quantization: q = (r / scale) + zero_point
        qp = input_details[0].get('quantization_parameters', {})
        if qp and len(qp.get('scales', [])) > 0:
            s = qp['scales'][0]
            zp = qp['zero_points'][0]
            blob = ((rgb.astype(np.float32) / 255.0) / s + zp)
            blob = np.clip(blob, -128, 127).astype(np.int8)
        else:
            blob = (rgb.astype(np.int32) - 128).astype(np.int8)
    else:
        blob = rgb.astype(np.float32) / 255.0

    blob = np.expand_dims(blob, axis=0)  # [1, 320, 320, 3] NHWC
    return blob, scale, pad_w, pad_h, w, h


# =============================================================================
#  POSTPROCESSING
# =============================================================================

def nms_numpy(boxes, scores, iou_threshold):
    """Fast NumPy NMS — no OpenCV/torchvision dependency"""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def postprocess(raw_output, scale, pad_w, pad_h, orig_w, orig_h):
    """
    Parse YOLOv8 TFLite output → list of detections.
    Handles both [1,C,N] and [1,N,C] output layouts.
    """
    # Dequantize output if needed
    out_detail = output_details[0]
    output = raw_output.copy()

    if out_detail['dtype'] != np.float32:
        qp = out_detail.get('quantization_parameters', {})
        if qp and len(qp.get('scales', [])) > 0:
            s = qp['scales'][0]
            zp = qp['zero_points'][0]
            output = (output.astype(np.float32) - zp) * s
        else:
            output = output.astype(np.float32)

    # Squeeze batch dimension → [C, N] or [N, C]
    output = output.squeeze(0)

    num_classes = len(CLASS_NAMES)
    expected_c = 4 + num_classes  # 6 for 2 classes

    # Determine layout and transpose if needed
    if output.ndim == 2:
        if output.shape[0] == expected_c and output.shape[1] != expected_c:
            output = output.T  # [C, N] → [N, C]
        elif output.shape[1] != expected_c and output.shape[0] != expected_c:
            # Unknown layout — try smaller dimension as channels
            if output.shape[0] < output.shape[1]:
                output = output.T

    # Now: [N, 4+num_classes]  where each row = [cx, cy, w, h, s0, s1]
    boxes_cxcywh = output[:, :4]                 # center-x, center-y, width, height
    class_scores = output[:, 4:4 + num_classes]  # class confidence scores

    # Best class per anchor
    max_scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)

    # Confidence filter
    mask = max_scores >= CONF_THRESHOLD
    boxes_cxcywh = boxes_cxcywh[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(boxes_cxcywh) == 0:
        return []

    # Convert center format → corner format
    x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

    # Remove letterbox padding and scale to original image coords
    x1 = (x1 - pad_w) / scale
    y1 = (y1 - pad_h) / scale
    x2 = (x2 - pad_w) / scale
    y2 = (y2 - pad_h) / scale

    # Clip to image bounds
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    # NMS
    xyxy = np.stack([x1, y1, x2, y2], axis=1)
    keep = nms_numpy(xyxy, max_scores, IOU_THRESHOLD)

    detections = []
    for i in keep:
        detections.append({
            "label": CLASS_NAMES[class_ids[i]],
            "confidence": float(max_scores[i]),
            "bbox": {
                "x1": float(x1[i]), "y1": float(y1[i]),
                "x2": float(x2[i]), "y2": float(y2[i])
            }
        })

    return detections


# =============================================================================
#  DETECTION (preprocess → infer → postprocess)
# =============================================================================

def run_detection(frame):
    """Full detection pipeline on a single frame"""
    blob, scale, pad_w, pad_h, orig_w, orig_h = preprocess(frame)

    # ── Inference ──
    t_infer = time.monotonic()
    interp.set_tensor(input_details[0]['index'], blob)
    interp.invoke()
    raw_output = interp.get_tensor(output_details[0]['index'])
    inference_ms = (time.monotonic() - t_infer) * 1000

    # ── Postprocess ──
    detections = postprocess(raw_output, scale, pad_w, pad_h, orig_w, orig_h)

    # ── Draw on frame ──
    annotated = frame.copy()
    for det in detections:
        b = det["bbox"]
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        conf = det["confidence"]
        label_name = det["label"]

        is_venomous = label_name == "venomous"
        color = (0, 0, 255) if is_venomous else (0, 255, 0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = f"{label_name} {conf:.0%}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, text, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    return detections, annotated, inference_ms


# =============================================================================
#  CAMERA
# =============================================================================

def init_camera():
    """Initialize Pi camera (rpicam-vid > OpenCV fallback)"""
    import subprocess

    # Try rpicam-vid first (Pi Camera Module)
    try:
        proc = subprocess.Popen(
            [
                "rpicam-vid",
                "--width", str(FRAME_WIDTH),
                "--height", str(FRAME_HEIGHT),
                "--framerate", str(CAMERA_FPS),
                "--codec", "mjpeg",
                "--quality", "70",
                "-t", "0",
                "-o", "-",
                "--nopreview",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        time.sleep(2)
        if proc.poll() is None:
            print(f"[+] Camera: rpicam-vid ({FRAME_WIDTH}x{FRAME_HEIGHT})")
            return ("rpicam", proc)
    except FileNotFoundError:
        pass

    # OpenCV fallback
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[+] Camera: OpenCV V4L2 ({FRAME_WIDTH}x{FRAME_HEIGHT})")
        return ("opencv", cap)

    print("[-] No camera — upload-only mode")
    return ("none", None)


def read_frame(cam_type, cam):
    """Read one frame from camera"""
    if cam_type == "rpicam":
        buf = b""
        while True:
            chunk = cam.stdout.read(4096)
            if not chunk:
                return None
            buf += chunk
            start = buf.find(b"\xff\xd8")
            end = buf.find(b"\xff\xd9", start + 2) if start >= 0 else -1
            if start >= 0 and end >= 0:
                jpg = buf[start:end + 2]
                return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    elif cam_type == "opencv":
        ret, frame = cam.read()
        return frame if ret else None
    return None


# =============================================================================
#  DETECTION LOOP (real FPS measurement)
# =============================================================================

def detection_loop():
    """Main loop: capture → preprocess → infer → postprocess → draw"""
    global latest_frame, latest_detections, latest_pipeline_fps, latest_inference_ms, running

    cam_type, cam = init_camera()
    if cam is None:
        print("[!] No camera — use 'Upload Image' to test detection")
        return

    fps_history = []
    infer_history = []
    frame_count = 0
    prev_time = time.monotonic()

    while running:
        frame = read_frame(cam_type, cam)
        if frame is None:
            time.sleep(0.01)
            continue

        # ── Full pipeline timing (capture-to-result) ──
        loop_start = time.monotonic()

        detections, annotated, infer_ms = run_detection(frame)

        loop_end = time.monotonic()
        loop_ms = (loop_end - loop_start) * 1000

        # Pipeline FPS = how fast the full loop actually runs (HONEST number)
        pipeline_fps = 1000.0 / max(loop_ms, 0.1)

        fps_history.append(pipeline_fps)
        infer_history.append(infer_ms)
        if len(fps_history) > 30:
            fps_history.pop(0)
            infer_history.pop(0)

        avg_fps = sum(fps_history) / len(fps_history)
        avg_infer = sum(infer_history) / len(infer_history)

        # Draw HONEST stats on frame
        cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(annotated, f"Infer: {avg_infer:.0f}ms", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(annotated, f"{model_dtype}", (10, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 130, 255), 1)

        # Danger banner
        is_danger = any(d["label"] == "venomous" for d in detections)
        if is_danger:
            h, w = annotated.shape[:2]
            cv2.rectangle(annotated, (0, h - 30), (w, h), (0, 0, 200), -1)
            cv2.putText(annotated, "!! VENOMOUS SNAKE !!", (10, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        with frame_lock:
            latest_frame = annotated
            latest_detections = detections
            latest_pipeline_fps = avg_fps
            latest_inference_ms = avg_infer

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  Frame {frame_count} | FPS: {avg_fps:.1f} | "
                  f"Infer: {avg_infer:.0f}ms | Dets: {len(detections)} | {model_dtype}")

    # Cleanup
    if cam_type == "rpicam" and cam.poll() is None:
        cam.terminate()
    elif cam_type == "opencv":
        cam.release()


# =============================================================================
#  MJPEG STREAM
# =============================================================================

def generate_mjpeg():
    """MJPEG stream generator for browser"""
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.1)
                continue
            _, jpeg = cv2.imencode('.jpg', latest_frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.03)


# =============================================================================
#  HTML UI
# =============================================================================

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Snake Detector — BIO_MIMIC</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0a0a0f;color:#e0e0e0;min-height:100vh}
.hdr{background:linear-gradient(135deg,#1a1a2e,#16213e);padding:18px 28px;border-bottom:2px solid #0f3460;display:flex;justify-content:space-between;align-items:center}
.hdr h1{font-size:1.4rem;background:linear-gradient(90deg,#a78bfa,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.badge{background:#0f3460;padding:5px 12px;border-radius:18px;font-size:.82rem;color:#53d769;border:1px solid #53d769}
.wrap{max-width:900px;margin:24px auto;padding:0 18px}
.vid{background:#111;border-radius:12px;overflow:hidden;border:2px solid #1a1a2e;box-shadow:0 8px 32px rgba(0,0,0,.4);min-height:200px}
.vid img{width:100%;display:block;min-height:240px;background:#111}
.no-cam{padding:50px 20px;text-align:center;color:#666;font-size:1.05em}
.no-cam b{color:#a78bfa}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:18px}
.sc{background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:10px;padding:14px;text-align:center;border:1px solid #0f3460}
.sc .v{font-size:1.7rem;font-weight:bold;color:#53d769}
.sc .l{font-size:.72rem;color:#888;margin-top:3px;text-transform:uppercase;letter-spacing:1px}
.sc.danger .v{color:#e94560}
.dets{margin-top:18px;background:#111;border-radius:10px;padding:14px;border:1px solid #1a1a2e}
.di{display:flex;justify-content:space-between;align-items:center;padding:9px 13px;border-radius:7px;margin-bottom:5px}
.di.venomous{background:rgba(233,69,96,.15);border-left:3px solid #e94560}
.di.non_venomous{background:rgba(83,215,105,.12);border-left:3px solid #53d769}
h3.sec{margin-top:28px;color:#888;font-size:.95rem}
.up{margin-top:10px;border:2px dashed #333;border-radius:12px;padding:28px;text-align:center;cursor:pointer;transition:all .3s;background:rgba(255,255,255,.02)}
.up:hover{border-color:#7c3aed;background:rgba(124,58,237,.05)}
#fi{display:none}
.ur{margin-top:14px;display:none;background:#111;border-radius:10px;overflow:hidden;border:1px solid #1a1a2e}
.ur.act{display:block}
.ur canvas{max-width:100%;display:block;margin:0 auto}
.sp{display:none;text-align:center;padding:18px}
.sp.act{display:block}
.sr{width:34px;height:34px;border:4px solid #222;border-top-color:#7c3aed;border-radius:50%;animation:sp .8s linear infinite;margin:0 auto 8px}
@keyframes sp{to{transform:rotate(360deg)}}
.ab{display:none;padding:13px 18px;border-radius:11px;margin-top:12px;font-weight:700;text-align:center}
.ab.danger{display:block;background:rgba(239,68,68,.15);border:2px solid #ef4444;color:#ef4444}
.ab.safe{display:block;background:rgba(34,197,94,.1);border:2px solid #22c55e;color:#22c55e}
.ft{margin-top:20px;text-align:center;color:#444;font-size:.73rem}
.eng{display:inline-block;background:#1a1a2e;border:1px solid #7c3aed;padding:3px 10px;border-radius:6px;color:#a78bfa;font-size:.72rem;margin-top:6px}
</style>
</head>
<body>
<div class="hdr">
<h1>🐍 Snake Detector — BIO_MIMIC</h1>
<span class="badge">● TFLite+XNNPACK</span>
</div>
<div class="wrap">
<div class="vid" id="vb">CAMERA_SLOT</div>
<div class="stats">
<div class="sc"><div class="v" id="fps">--</div><div class="l">Pipeline FPS</div></div>
<div class="sc"><div class="v" id="infer">--</div><div class="l">Infer ms</div></div>
<div class="sc"><div class="v" id="dc">0</div><div class="l">Detections</div></div>
<div class="sc" id="dcard"><div class="v" id="ds">SAFE</div><div class="l">Threat</div></div>
</div>
<div class="dets" id="dl"><p style="color:#555;text-align:center">Waiting...</p></div>
<h3 class="sec">📤 Test with Image</h3>
<div class="up" id="uz">📷 Click or drag a snake image here</div>
<input type="file" id="fi" accept="image/*">
<div class="sp" id="sp"><div class="sr"></div><div>Detecting...</div></div>
<div id="ab" class="ab"></div>
<div class="ur" id="ur"><canvas id="cv"></canvas><div id="udi" style="padding:12px"></div></div>
<div class="ft">
YOLOv8n + TFLite + XNNPACK | Raspberry Pi 4 | ARM64
<br><span class="eng" id="eng">--</span>
</div>
</div>
<script>
function us(){fetch('/api/status').then(r=>r.json()).then(d=>{
document.getElementById('fps').textContent=d.pipeline_fps.toFixed(1);
document.getElementById('infer').textContent=d.inference_ms.toFixed(0);
document.getElementById('dc').textContent=d.detections.length;
document.getElementById('eng').textContent=d.engine;
let h=d.detections.some(x=>x.label==='venomous');
let c=document.getElementById('dcard'),s=document.getElementById('ds');
if(h){s.textContent='DANGER';c.classList.add('danger')}else{s.textContent='SAFE';c.classList.remove('danger')}
let l=document.getElementById('dl');
if(!d.detections.length){l.innerHTML='<p style="color:#555;text-align:center">No snakes detected</p>'}
else{l.innerHTML=d.detections.map(x=>'<div class="di '+x.label+'"><span>'+(x.label==='venomous'?'🔴':'🟢')+' '+x.label+'</span><span>'+(x.confidence*100).toFixed(0)+'%</span></div>').join('')}
}).catch(()=>{})}
setInterval(us,500);
let uz=document.getElementById('uz'),fi=document.getElementById('fi'),cv=document.getElementById('cv'),cx=cv.getContext('2d');
uz.onclick=()=>fi.click();
uz.ondragover=e=>{e.preventDefault();uz.style.borderColor='#7c3aed'};
uz.ondragleave=()=>uz.style.borderColor='#333';
uz.ondrop=e=>{e.preventDefault();uz.style.borderColor='#333';if(e.dataTransfer.files.length)uf(e.dataTransfer.files[0])};
fi.onchange=e=>{if(e.target.files.length)uf(e.target.files[0])};
function uf(f){let sp=document.getElementById('sp'),ur=document.getElementById('ur'),ab=document.getElementById('ab');
ur.classList.remove('act');ab.style.display='none';sp.classList.add('act');
let fd=new FormData();fd.append('image',f);
fetch('/api/detect_image',{method:'POST',body:fd}).then(r=>r.json()).then(d=>{
sp.classList.remove('act');
if(d.result_image){let img=new Image();img.onload=()=>{let s=Math.min(860/img.width,600/img.height,1);
cv.width=img.width*s;cv.height=img.height*s;cx.drawImage(img,0,0,cv.width,cv.height);ur.classList.add('act')};
img.src='data:image/jpeg;base64,'+d.result_image}
let it=document.getElementById('udi'),hv=d.detections.some(x=>x.label==='venomous');
if(!d.detections.length){it.innerHTML='<p style="color:#666;text-align:center">No snakes detected</p>';ab.style.display='none'}
else{it.innerHTML=d.detections.map(x=>'<div class="di '+x.label+'"><span>'+(x.label==='venomous'?'🔴':'🟢')+' '+x.label+'</span><span>'+(x.confidence*100).toFixed(0)+'%</span></div>').join('');
if(hv){ab.className='ab danger';ab.textContent='⚠️ VENOMOUS SNAKE DETECTED!'}else{ab.className='ab safe';ab.textContent='✅ Non-venomous — Safe'}}
}).catch(e=>{sp.classList.remove('act');alert('Error: '+e.message)})}
</script>
</body>
</html>"""


# =============================================================================
#  ROUTES
# =============================================================================

@app.route('/')
def index():
    has_cam = latest_frame is not None or running
    if has_cam:
        cam_html = '<img src="/video_feed" alt="Live Feed" />'
    else:
        cam_html = '<div class="no-cam">📷 No camera<br>Use <b>Upload Image</b> below</div>'
    return HTML.replace('CAMERA_SLOT', cam_html)


@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def api_status():
    with frame_lock:
        return jsonify({
            "pipeline_fps": latest_pipeline_fps,
            "inference_ms": latest_inference_ms,
            "detections": latest_detections,
            "engine": f"TFLite {model_dtype} + XNNPACK | {model_name}",
            "model": model_name,
            "dtype": model_dtype
        })


@app.route('/api/detect_image', methods=['POST'])
def api_detect_image():
    """Upload image for detection"""
    if 'image' not in request.files:
        return jsonify({"error": "No image"}), 400
    img_bytes = request.files['image'].read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    detections, annotated, infer_ms = run_detection(img)
    _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jsonify({
        "detections": detections,
        "inference_ms": infer_ms,
        "result_image": base64.b64encode(buf).decode('utf-8')
    })


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 55)
    print(" 🐍 Snake Detector — TFLite + XNNPACK + INT8")
    print("=" * 55)
    print(f"  Model dir:  {MODEL_DIR}")
    print(f"  Image size: {IMG_SIZE}")
    print(f"  Threads:    {NUM_THREADS}")
    print(f"  Conf:       {CONF_THRESHOLD}")
    print(f"  Runtime:    {TFLITE_SOURCE}")
    print()

    load_model()

    det_thread = threading.Thread(target=detection_loop, daemon=True)
    det_thread.start()

    ip = "0.0.0.0"
    port = 5000
    print(f"[+] Web UI: http://{ip}:{port}")
    app.run(host=ip, port=port, debug=False, threaded=True)
