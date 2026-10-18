
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import pytesseract


DEFAULT_TESSERACT_PATH = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
configured_tesseract = os.environ.get("TESSERACT_CMD")
if configured_tesseract:
    pytesseract.pytesseract.tesseract_cmd = configured_tesseract
elif DEFAULT_TESSERACT_PATH.exists():
    pytesseract.pytesseract.tesseract_cmd = str(DEFAULT_TESSERACT_PATH)


def _read_text_and_confidence(image):
    data = pytesseract.image_to_data(
        image,
        config="--oem 3 --psm 7",
        output_type=pytesseract.Output.DICT,
    )
    words = [
        (text.strip(), float(confidence))
        for text, confidence in zip(data["text"], data["conf"])
        if text.strip() and float(confidence) >= 0
    ]
    text = " ".join(word for word, _ in words).upper()
    confidence = sum(score for _, score in words) / len(words) if words else 0.0
    return text, confidence


def read_bus_sign(
    image,
    save_crops=False,
    region=None,
    vertical_band=(0.06, 0.20),
    route_horizontal=(0.17, 0.38),
    destination_horizontal=(0.32, 0.82),
):
    if image is None or image.size == 0:
        raise ValueError("A non-empty bus image is required for OCR.")

    sign_image = image
    if region is not None:
        image_height, image_width = image.shape[:2]
        x1, y1, x2, y2 = map(int, region)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("The detected bus region is invalid.")
        padding_x = int((x2 - x1) * 0.10)
        padding_y = int((y2 - y1) * 0.10)
        left = max(0, x1 - padding_x)
        top = max(0, y1 - padding_y)
        right = min(image_width, x2 + padding_x)
        bottom = min(image_height, y2 + padding_y)
        sign_image = image[top:bottom, left:right]
        if sign_image.size == 0:
            raise ValueError("The detected bus region is outside the camera image.")

    height, width = sign_image.shape[:2]
    band_top, band_bottom = vertical_band
    if not 0 <= band_top < band_bottom <= 1:
        raise ValueError("The OCR vertical band must be within 0 and 1.")
    route_crop = sign_image[
        int(height * band_top):int(height * band_bottom),
        int(width * route_horizontal[0]):int(width * route_horizontal[1])
    ]
    destination_crop = sign_image[
        int(height * band_top):int(height * band_bottom),
        int(width * destination_horizontal[0]):int(
            width * destination_horizontal[1]
        )
    ]

    if route_crop.size == 0 or destination_crop.size == 0:
        raise ValueError("The bus image is too small to contain the sign crops.")
    if save_crops:
        cv2.imwrite("route_crop.png", route_crop)
        cv2.imwrite("destination_crop.png", destination_crop)

    route_gray = cv2.cvtColor(route_crop, cv2.COLOR_BGR2GRAY)
    route_gray = cv2.resize(
        route_gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
    )
    destination_gray = cv2.cvtColor(destination_crop, cv2.COLOR_BGR2GRAY)
    destination_gray = cv2.resize(
        destination_gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        route_future = executor.submit(_read_text_and_confidence, route_gray)
        destination_future = executor.submit(
            _read_text_and_confidence, destination_gray
        )
        route_text, route_confidence = route_future.result()
        destination_text, destination_confidence = destination_future.result()
    return {
        "route_text": route_text,
        "route_confidence": route_confidence,
        "destination_text": destination_text,
        "destination_confidence": destination_confidence,
    }


def main():
    image_path = Path(__file__).resolve().parents[1] / "bus.png"
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Could not read sample image: {image_path}")

    try:
        reading = read_bus_sign(image, save_crops=True)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"OCR failed: {error}") from error

    print("OCR Route Text:", repr(reading["route_text"]))
    print("Route Confidence:", round(reading["route_confidence"], 1))
    print("OCR Destination Text:", repr(reading["destination_text"]))
    print("Destination Confidence:", round(reading["destination_confidence"], 1))


if __name__ == "__main__":
    main()