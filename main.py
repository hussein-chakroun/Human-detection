from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 model (use a smaller model for faster inference)
model = YOLO("yolov8x.pt")  # Use yolov8s (small) for speed

# Initialize webcam (use index 1 and CAP_DSHOW backend)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Change index to 0 if 1 doesn't work

# Function to enhance low-light images while preserving color
def enhance_low_light(frame):
    # Convert the frame to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image into L, A, and B channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L channel (lightness)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge the enhanced L channel back with A and B channels
    enhanced_lab = cv2.merge((l, a, b))
    
    # Convert the LAB image back to BGR color space
    enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.CAB2BGR)
    
    return enhanced_frame

# Frame skipping counter
frame_counter = 0
skip_frames = 2  # Process every 3rd frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames to reduce lag
    frame_counter += 1
    if frame_counter % skip_frames != 0:
        continue

    # Resize frame to lower resolution for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Enhance low-light images while preserving color (optional)
    # frame = enhance_low_light(frame)

    # Perform human detection with higher confidence threshold
    results = model.predict(frame, conf=0.5, classes=0)  # Detect only humans (class 0)

    # Visualize results
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("Human Detection", annotated_frame)

    # Exit on 'q' key press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()