import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened!")
    exit()

print("Bus detection started!")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Could not read camera frame!")
        break

    # Run YOLO detection
    results = model(frame, verbose=False)

    # Create a copy of the original frame
    bus_frame = frame.copy()

    for result in results:
        for box in result.boxes:

            # Get class ID
            class_id = int(box.cls[0])

            # Get class name
            class_name = model.names[class_id]

            # Keep only buses
            if class_name == "bus":

                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get confidence
                confidence = float(box.conf[0])

                # Draw bounding box
                cv2.rectangle(
                    bus_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Display label
                label = f"BUS {confidence:.2f}"

                cv2.putText(
                    bus_frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

    # Display camera
    cv2.imshow("Bus Detection", bus_frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()