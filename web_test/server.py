"""
Snake Detector — Test Server
Run: python server.py
Open: http://localhost:5000
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import os, base64, cv2
import numpy as np

app = Flask(__name__, static_folder=".")
CORS(app)

# ── Venomous species list (lowercase, partial match friendly) ──────────────
VENOMOUS_KEYWORDS = [
    "cobra", "krait", "viper", "rattlesnake", "mamba", "taipan",
    "copperhead", "cottonmouth", "water moccasin", "coral snake",
    "boomslang", "fer-de-lance", "bushmaster", "death adder",
    "russell", "saw-scaled", "pit viper", "king cobra"
]

def is_venomous(name: str) -> bool:
    name_lower = name.lower()
    return any(kw in name_lower for kw in VENOMOUS_KEYWORDS)

# ── Model loading ─────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "snake_model_for_pi", "best.pt")
MODEL_PATH = os.path.abspath(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found: {MODEL_PATH}")
    exit(1)

print(f"Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH, task="detect")

# ── Print all class names on startup so you can verify mapping ────────────
print("✅ Model loaded!")
print("📋 Model class names:")
for idx, name in model.names.items():
    tag = "🔴 VENOMOUS" if is_venomous(name) else "🟢 non-venomous"
    print(f"   [{idx:>3}] {name:<40} {tag}")

# ── Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/detect", methods=["POST"])
def detect():
    data = request.json
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        img_b64 = data["image"].split(",")[-1]
        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Could not decode image"}), 400

        # Maximized sensitivity (low conf + augment) but strictly limited to 1 box
        results = model.predict(img, imgsz=640, conf=0.05, iou=0.1, augment=True, max_det=1, verbose=False)

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                cls  = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                name = model.names[cls]
                venomous = is_venomous(name)

                print(f"  Detected: {name} (cls={cls}, conf={conf:.2f}, venomous={venomous})")

                detections.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "class": cls,
                    "name": name,
                    "confidence": conf,
                    "venomous": venomous
                })

        return jsonify({"detections": detections})

    except Exception as e:
        print(f"❌ Detection error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🐍 Snake Detector running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)