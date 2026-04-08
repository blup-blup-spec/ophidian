from ultralytics import YOLO
import os

model_path = r"c:\IOT PROJECTS\BIO_MIMIC\snake_model_for_pi\best.pt"
if not os.path.exists(model_path):
    print(f"Model not found: {model_path}")
else:
    try:
        model = YOLO(model_path)
        print("Model loaded successfully!")
        print("Class names:", model.names)
    except Exception as e:
        print(f"Error loading model: {e}")
