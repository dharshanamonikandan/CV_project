import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened!")
    exit()

print("Approaching detection started!")

previous_area = None
increasing_count = 0

while True:
    ret, frame = camera.read()

    if not ret:
        print("Could not read camera frame!")
        break

    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]

    # Detection zone
    zone_top = int(frame_height * 0.35)
    zone_bottom = int(frame_height * 0.90)
    zone_left = int(frame_width * 0.15)
    zone_right = int(frame_width * 0.85)

    approaching = False

    # Run YOLO
    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # Only process buses
            if class_name == "bus":

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                width = x2 - x1
                height = y2 - y1
                area = width * height

                confidence = float(box.conf[0])

                print("Bus area:", area)

                # Compare area with previous frame
                if previous_area is not None:

                    if area > previous_area * 1.01:
                        increasing_count += 1

                    elif area < previous_area * 0.99:
                        increasing_count = 0

                previous_area = area

                # Check whether bus is inside detection zone
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                inside_zone = (
                    zone_left <= center_x <= zone_right
                    and zone_top <= center_y <= zone_bottom
                )

                # Bus approaching only when:
                # 1. Area is increasing
                # 2. Bus is inside detection zone
                if increasing_count >= 5 and inside_zone:
                    approaching = True

                # Draw bus bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Bus label
                label = f"BUS | Area: {area}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    # Draw detection zone
    cv2.rectangle(
        frame,
        (zone_left, zone_top),
        (zone_right, zone_bottom),
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        "DETECTION ZONE",
        (zone_left, zone_top - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    # Display approaching status
    if approaching:
        status = "BUS APPROACHING"
    else:
        status = "Monitoring..."

    cv2.putText(
        frame,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show camera
    cv2.imshow("Approaching Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()