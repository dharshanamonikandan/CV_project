import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

plate = cv2.imread("plate_crop.png")

if plate is None:
    print("Error: plate_crop.png not found!")
else:
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # Convert to black text on a white background
    gray = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    # Add white space around the plate
    gray = cv2.copyMakeBorder(
        gray, 40, 40, 40, 40,
        cv2.BORDER_CONSTANT, value=255
    )

    text = pytesseract.image_to_string(
        gray,
        config="--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    print("Detected Bus Number:")
    print(text.strip())