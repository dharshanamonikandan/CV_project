import argparse
from collections import deque
from datetime import datetime
from pathlib import Path
import time

import cv2
from ultralytics import YOLO

from route_ocr_test import read_bus_sign
from validate_bus import find_verified_bus, load_bus_database


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = PROJECT_ROOT / "yolo11n.pt"
ANNOUNCEMENT_CONFIRMATIONS = 2
ANNOUNCEMENT_HISTORY_SIZE = 3
DEPARTURE_DELAY_SECONDS = 2.5
SIGN_READ_INTERVAL_SECONDS = 0.25
MOTION_SAMPLE_INTERVAL_SECONDS = 0.25
MOTION_HISTORY_SIZE = 5
MOTION_AREA_CHANGE_THRESHOLD = 0.05


def _classify_area_trend(area_history):
    if len(area_history) < MOTION_HISTORY_SIZE:
        return "DETECTING DIRECTION"

    first_area = area_history[0]
    latest_area = area_history[-1]
    if first_area <= 0:
        return "MOVEMENT UNCERTAIN"

    area_change = (latest_area - first_area) / first_area
    if area_change >= MOTION_AREA_CHANGE_THRESHOLD:
        return "APPROACHING"
    if area_change <= -MOTION_AREA_CHANGE_THRESHOLD:
        return "LEAVING"
    return "MOVEMENT UNCERTAIN"


def _should_read_sign(inside_zone, bus_verified, now, last_ocr_time):
    return (
        inside_zone
        and not bus_verified
        and now - last_ocr_time >= SIGN_READ_INTERVAL_SECONDS
    )


def _record_bus_candidate(recent_matches, candidate_bus):
    if candidate_bus is None:
        recent_matches.append((None, None))
        return None

    bus_key = (
        candidate_bus["route_number"].upper(),
        candidate_bus["destination"].upper(),
    )
    recent_matches.append((bus_key, candidate_bus))
    confirmations = sum(key == bus_key for key, _ in recent_matches)
    if confirmations >= ANNOUNCEMENT_CONFIRMATIONS:
        return candidate_bus
    return None


def _read_and_verify_sign(image, region):
    reading = read_bus_sign(image, region=region)
    verified_bus = find_verified_bus(
        reading["route_text"],
        reading["destination_text"],
        reading["route_confidence"],
        reading["destination_confidence"],
    )
    if verified_bus is not None:
        return reading, verified_bus

    fallback_reading = read_bus_sign(
        image,
        region=region,
        vertical_band=(0.26, 0.42),
        destination_horizontal=(0.32, 0.88),
    )
    fallback_bus = find_verified_bus(
        fallback_reading["route_text"],
        fallback_reading["destination_text"],
        fallback_reading["route_confidence"],
        fallback_reading["destination_confidence"],
    )
    if fallback_bus is not None:
        return fallback_reading, fallback_bus

    return reading, None


class VoiceAnnouncer:
    def __init__(self):
        self.engine = None
        try:
            import pyttsx3

            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 165)
        except Exception as error:
            print(f"Voice output is unavailable: {error}")

    def announce(self, bus):
        if self.engine is None:
            return False

        message = (
            f"Attention. Bus route {bus['route_number']} to "
            f"{bus['destination']} is approaching."
        )
        try:
            self.engine.say(message)
            self.engine.runAndWait()
            return True
        except Exception as error:
            print(f"Voice announcement failed: {error}")
            self.engine = None
            return False


def _get_bus_detections(results):
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                "box": (x1, y1, x2, y2),
                "area": max(0, x2 - x1) * max(0, y2 - y1),
                "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                "confidence": float(box.conf[0]),
            })
    return detections


def _draw_departures(frame, departures):
    cv2.putText(
        frame, "Recently departed:", (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
    )
    for index, (route, destination, time_text) in enumerate(departures):
        label = f"{time_text}  {route} to {destination}"
        cv2.putText(
            frame, label, (20, 140 + index * 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )


def _draw_bus_status(frame, direction, inside_zone, verified_bus, reading=None):
    if inside_zone or verified_bus is not None:
        direction_text = f"Direction (estimate): {direction.replace('_', ' ')}"
    else:
        direction_text = "Direction (estimate): NO BUS IN ZONE"

    if direction == "APPROACHING":
        direction_color = (0, 220, 80)
    elif direction == "LEAVING":
        direction_color = (0, 165, 255)
    else:
        direction_color = (0, 220, 255)

    if verified_bus is None:
        if reading is not None:
            route_ocr = reading.get("route_text", "").strip() or "not read"
            destination_ocr = (
                reading.get("destination_text", "").strip() or "not read"
            )
            route_confidence = reading.get("route_confidence", 0.0)
            destination_confidence = reading.get("destination_confidence", 0.0)
            bus_number_text = (
                f"Bus number: {route_ocr[:18]} (unverified, "
                f"{route_confidence:.0f}%)"
            )
            route_text = (
                f"Route: {destination_ocr[:28]} (unverified, "
                f"{destination_confidence:.0f}%)"
            )
        else:
            bus_number_text = "Bus number: waiting for OCR"
            route_text = "Route: waiting for OCR"
    else:
        bus_number_text = f"Bus number: {verified_bus['route_number']}"
        origin = verified_bus.get("origin", "").strip()
        route_destination = verified_bus["destination"]
        if origin:
            route_destination = f"{origin} to {route_destination}"
        route_text = f"Route: {route_destination}"

    for text, y, color in (
        (direction_text, 40, direction_color),
        (bus_number_text, 70, (255, 255, 255)),
        (route_text, 100, (255, 255, 255)),
    ):
        cv2.putText(
            frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, color, 2,
        )


def run(camera_index, model_path, confidence):
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    load_bus_database()
    model = YOLO(str(model_path))
    bus_class_id = next(
        (class_id for class_id, name in model.names.items() if name.lower() == "bus"),
        None,
    )
    if bus_class_id is None:
        raise RuntimeError("The selected YOLO model has no 'bus' class.")

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open camera {camera_index}. Check that it is connected "
            "and not being used by another application."
        )

    announcer = VoiceAnnouncer()
    recent_departures = deque(maxlen=3)
    recent_matches = deque(maxlen=ANNOUNCEMENT_HISTORY_SIZE)
    area_history = deque(maxlen=MOTION_HISTORY_SIZE)
    last_motion_sample_time = 0.0
    direction = "UNKNOWN"
    last_bus_seen_time = 0.0
    last_ocr_time = 0.0
    verified_bus = None
    announced_bus_key = None
    last_reading = None

    print("Smart Bus Identification is running. Press Q to quit.")
    try:
        while True:
            frame_read, frame = camera.read()
            if not frame_read:
                print("Could not read a frame from the camera; stopping.")
                break
            ocr_frame = frame.copy()

            frame_height, frame_width = frame.shape[:2]
            zone_top = int(frame_height * 0.35)
            zone_bottom = int(frame_height * 0.90)
            zone_left = int(frame_width * 0.15)
            zone_right = int(frame_width * 0.85)

            results = model(
                frame,
                classes=[bus_class_id],
                conf=confidence,
                verbose=False,
            )
            detections = _get_bus_detections(results)
            for detection in detections:
                x1, y1, x2, y2 = detection["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 80), 2)
                cv2.putText(
                    frame,
                    f"BUS {detection['confidence']:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 220, 80),
                    2,
                )

            cv2.rectangle(
                frame,
                (zone_left, zone_top),
                (zone_right, zone_bottom),
                (255, 120, 0),
                2,
            )
            inside_zone = [
                detection for detection in detections
                if zone_left <= detection["center"][0] <= zone_right
                and zone_top <= detection["center"][1] <= zone_bottom
            ]
            now = time.monotonic()
            if detections:
                last_bus_seen_time = now
            if inside_zone:
                target = max(inside_zone, key=lambda detection: detection["area"])
                if now - last_motion_sample_time >= MOTION_SAMPLE_INTERVAL_SECONDS:
                    area_history.append(target["area"])
                    last_motion_sample_time = now
                direction = _classify_area_trend(area_history)
            else:
                area_history.clear()
                last_motion_sample_time = 0.0
                if not detections and verified_bus is None:
                    recent_matches.clear()
                    direction = "UNKNOWN"

            if _should_read_sign(
                bool(detections), verified_bus is not None, now, last_ocr_time
            ):
                last_ocr_time = now
                ocr_target = max(
                    inside_zone or detections,
                    key=lambda detection: detection["area"],
                )
                try:
                    reading, candidate_bus = _read_and_verify_sign(
                        ocr_frame, ocr_target["box"]
                    )
                    last_reading = reading
                except Exception as error:
                    candidate_bus = None
                    last_reading = None
                    print(f"Sign OCR failed: {error}")

                confirmed_bus = _record_bus_candidate(
                    recent_matches, candidate_bus
                )
                if confirmed_bus is not None:
                    verified_bus = confirmed_bus

            if (
                verified_bus is not None
                and direction == "APPROACHING"
                and announced_bus_key != (
                    verified_bus["route_number"].upper(),
                    verified_bus["destination"].upper(),
                )
            ):
                announced_bus_key = (
                    verified_bus["route_number"].upper(),
                    verified_bus["destination"].upper(),
                )
                announced = announcer.announce(verified_bus)
                if announced:
                    print(
                        "Announced verified approaching bus: "
                        f"{verified_bus['route_number']} to "
                        f"{verified_bus['destination']}"
                    )
                else:
                    print(
                        "Bus verified, but voice output is unavailable: "
                        f"{verified_bus['route_number']} to "
                        f"{verified_bus['destination']}"
                    )

            if (
                verified_bus is not None
                and now - last_bus_seen_time > DEPARTURE_DELAY_SECONDS
            ):
                departure_time = datetime.now().strftime("%H:%M")
                recent_departures.appendleft((
                    verified_bus["route_number"],
                    verified_bus["destination"],
                    departure_time,
                ))
                print(
                    f"Bus departed: {verified_bus['route_number']} to "
                    f"{verified_bus['destination']} at {departure_time}"
                )
                verified_bus = None
                announced_bus_key = None
                recent_matches.clear()
                direction = "UNKNOWN"
                last_reading = None

            _draw_bus_status(
                frame, direction, bool(detections), verified_bus, last_reading
            )
            _draw_departures(frame, recent_departures)
            cv2.imshow("Smart Bus Identification", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Detect approaching buses, verify route signs, and announce routes."
    )
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument(
        "--model", type=Path, default=MODEL_FILE,
        help="path to YOLO weights (default: project yolo11n.pt)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.35,
        help="minimum YOLO bus detection confidence (0 to 1)",
    )
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")

    try:
        run(args.camera, args.model, args.confidence)
    except Exception as error:
        parser.exit(1, f"Startup failed: {error}\n")


if __name__ == "__main__":
    main()