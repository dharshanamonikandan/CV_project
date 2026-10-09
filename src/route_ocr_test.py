
import cv2
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Read bus image
image = cv2.imread("bus.png")

if image is None:
    print("Error: bus.png not found!")
    raise SystemExit

h, w = image.shape[:2]

# 1. Crop route number area
route_crop = image[
    int(h * 0.06):int(h * 0.20),
    int(w * 0.17):int(w * 0.38)
]

cv2.imwrite("route_crop.png", route_crop)

route_gray = cv2.cvtColor(
    route_crop, cv2.COLOR_BGR2GRAY
)

route_gray = cv2.resize(
    route_gray, None, fx=3, fy=3,
    interpolation=cv2.INTER_CUBIC
)

# Read route text automatically
route_text = pytesseract.image_to_string(
    route_gray, config="--oem 3 --psm 7"
).upper().strip()

print("OCR Route Text:", repr(route_text))

route_text = route_text.replace("!", "I").replace("|", "I")

match = re.search(r'(\d{1,3})\s*([A-Z])', route_text)

if match:
    detected_route = match.group(1) + match.group(2)
    print("Detected Route Number:", detected_route)
else:
    print("Route number not detected.")
    print("Please check route_crop.png")

# Extract route number from OCR text
pattern = r'\b\d{1,3}\s*[A-Z]\b'
route_match = re.search(pattern, route_text)

# 2. Crop destination display
dest_crop = image[
    int(h * 0.06):int(h * 0.20),
    int(w * 0.34):int(w * 0.82)
]

cv2.imwrite("destination_crop.png", dest_crop)

dest_gray = cv2.cvtColor(
    dest_crop, cv2.COLOR_BGR2GRAY
)

dest_gray = cv2.resize(
    dest_gray, None, fx=4, fy=4,
    interpolation=cv2.INTER_CUBIC
)

# Read destination text automatically
dest_text = pytesseract.image_to_string(
    dest_gray,
    config="--oem 3 --psm 7"
).upper().strip()

dest_text = dest_text.replace("|", "")

print("OCR Destination Text:", dest_text)