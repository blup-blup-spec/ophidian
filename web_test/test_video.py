import cv2
from ultralytics import YOLO
import argparse
import os

# Set up argument parsing for webcam vs video file
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='0', help='0 for webcam, or path to a video file (.mp4)')
args = parser.parse_args()

# Load the trained model
# Using relative path from web_test to where best.pt is saved
model_path = os.path.join(os.path.dirname(__file__), "..", "snake_model_for_pi", "best.pt")
model_path = os.path.abspath(model_path)

print(f"Loading YOLO model from: {model_path}")
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Open video source
# If source is "0", convert to integer for webcam, else keep as string for file path
source = int(args.source) if args.source.isdigit() else args.source
cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print(f"❌ Error: Could not open video source: {args.source}")
    exit(1)

print("\n✅ Video feed started!")
print("Press 'q' to quit out of the window.\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video feed ended or cannot read frame.")
        break
    
    # Run YOLOv8 inference on the frame
    results = model.predict(frame, conf=0.3, imgsz=320, verbose=False)
    
    # Ultralytics has a built-in annotator to draw the boxes quickly
    annotated_frame = results[0].plot()
    
    # Add a custom alert if a venomous snake is detected
    has_venomous = False
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            name = model.names[cls_id]
            if "venomous" in name.lower() and "non" not in name.lower():
                has_venomous = True
                
    if has_venomous:
        # Draw red warning text
        cv2.putText(annotated_frame, "!! VENOMOUS SNAKE DETECTED !!", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
    
    # Show the frame
    cv2.imshow("Snake Detector — Live Video Test", annotated_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
