import cv2
from ultralytics import YOLO

# Load YOLOv8 pre-trained model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Object Detection Started...")
print("Press 'q' or 'Esc' to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Detect and track objects
    results = model.track(frame, persist=True)

    # Draw boxes, labels and tracking IDs
    annotated_frame = results[0].plot()

    # Display output
    cv2.imshow("Object Detection and Tracking", annotated_frame)

    # Press q or Esc to stop
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()