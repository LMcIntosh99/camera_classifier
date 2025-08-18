import cv2
from ultralytics import YOLO

# Load a pretrained YOLOv8 model (small version, faster for realtime)
model = YOLO("yolov8n.pt")

# Open webcam
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame, stream=True, verbose=False)

    for r in results:
        for box in r.boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = model.names[class_id]

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    # Show video with detections
    cv2.imshow("YOLOv8 Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
