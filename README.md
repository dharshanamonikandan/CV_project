# AI-Based Smart Bus Identification

This project uses a webcam, YOLO bus detection, Tesseract OCR, a local route database, and text-to-speech to identify and announce verified approaching buses. It is a prototype: always test it in a safe, controlled setting and do not rely on it as the sole source of travel information.

## Requirements

- Windows with the project virtual environment at `venv\`.
- A webcam and speakers or headphones.
- Tesseract OCR installed. The default executable path is `C:\Program Files\Tesseract-OCR\tesseract.exe`.
- Python packages: OpenCV, Ultralytics, pytesseract, and pyttsx3. They are already available in the current project environment.

If packages need to be installed, open PowerShell in the project folder and run:

```powershell
.\venv\Scripts\python.exe -m pip install opencv-python ultralytics pytesseract pyttsx3
```

If Tesseract is installed somewhere else, set its executable path in that PowerShell window:

```powershell
$env:TESSERACT_CMD = "C:\Path\To\tesseract.exe"
```

## Run

From the project folder, run the offline tests first:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s src -p "test_*.py" -v
```

Start the webcam application:

```powershell
.\venv\Scripts\python.exe src\main.py
```

Press `Q` in the video window to quit. To select another webcam, for example camera 1:

```powershell
.\venv\Scripts\python.exe src\main.py --camera 1
```

The model defaults to the included `yolo11n.pt`. Use `--model` to choose another compatible YOLO model, and `--confidence 0.35` to adjust the minimum bus detection confidence (valid range: 0 to 1).

## Verification And Announcements

The application detects the largest bus in the existing central detection zone and compares five time-spaced bounding-box area samples. A total area increase of at least 5% indicates an approaching bus; a decrease of at least 5% indicates a leaving bus. Sign OCR runs whenever YOLO detects a bus, even before direction is known or the bus enters the movement-analysis zone. The route and destination crop fractions are applied to the bus detection region with 10% padding. If the usual upper sign crop does not verify, OCR tries a lower band and a wider destination crop to handle signs positioned lower in the detection box or destinations that extend farther across it. OCR reads the two crops in parallel and retries as soon as the previous read completes. Direction is an estimate based on apparent size, not reliable tracking of a bus moving sideways or changing lanes. The display shows direction, bus number, and route (origin to destination) on separate lines. OCR confidence must be at least 20 for the route and 50 for the destination. The route and destination must match one record in `src\bus_database.json`; the destination allows a small OCR difference, and a route ending in a letter allows one extra commonly confused `I`, `L`, or `1`. Two matching database-verified reads among the latest three OCR attempts are required before voice announcement. Uncertain or unlisted buses are not announced, and leaving buses are not announced.

The spoken message names the route and destination. A bus is announced at most once during one visit. After it has been absent from the detection zone for 2.5 seconds, the application marks it as departed and shows up to three recently departed, verified buses on screen. This is an estimate based on camera detection, not GPS or confirmed physical departure. The current detector follows one largest bus at a time.

If speech initialization or playback fails, the application reports the problem and continues without claiming that speech played. If OCR or the database cannot be read, the bus is not verified and is not announced.

## Add Routes

Edit the `buses` list in `src\bus_database.json`. Add one entry per route and destination; the same route number may have multiple destinations:

```json
{
	"buses": [
		{
			"route_number": "21A",
			"origin": "MADURAI",
			"destination": "THIRUMANGALAM"
		},
		{
			"route_number": "22B",
			"origin": "MADURAI",
			"destination": "PERIYAKULAM"
		}
	]
}
```

Use the destination wording shown on the bus sign. The validator normalizes letter case and punctuation but should not be expected to correct substantial OCR errors.

## Checks And Limitations

The test suite uses the included `bus.png` fixture and does not need a webcam. The webcam, live bus detection, approach/departure timing, and audible output still need to be checked on the target computer with real footage. Detection quality depends on camera placement, lighting, sign visibility, and the supplied YOLO model. Tesseract must be installed separately from the Python packages.
