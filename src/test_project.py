import json
import os
from collections import deque
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import validate_bus
from main import (
    MOTION_HISTORY_SIZE,
    _classify_area_trend,
    _draw_bus_status,
    _read_and_verify_sign,
    _record_bus_candidate,
    _should_read_sign,
)


class BusValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "bus_database.json"
        self.database_patch = patch.object(
            validate_bus, "DATABASE_FILE", self.database_path
        )
        self.database_patch.start()

    def tearDown(self):
        self.database_patch.stop()
        self.temp_dir.cleanup()

    def write_database(self, buses):
        self.database_path.write_text(
            json.dumps({"buses": buses}), encoding="utf-8"
        )

    def test_supports_multiple_routes_and_destinations(self):
        self.write_database([
            {"route_number": "21A", "destination": "THIRUMANGALAM"},
            {"route_number": "21A", "destination": "PERIYAKULAM"},
            {"route_number": "22B", "destination": "THIRUMANGALAM"},
        ])

        match = validate_bus.find_verified_bus(
            "21A", "MADURAI > PERIYAKULAM", 90, 90
        )
        self.assertEqual(match["destination"], "PERIYAKULAM")
        self.assertEqual(
            validate_bus.find_verified_bus(
                "22B", "MADURAI > THIRUMANGALAM", 90, 90
            )["route_number"],
            "22B",
        )

    def test_accepts_sample_ocr_confusions(self):
        self.write_database([
            {"route_number": "21A", "destination": "THIRUMANGALAM"}
        ])

        match = validate_bus.find_verified_bus(
            "21AL!", "| MADURAI > THIRUMANGALAN", 30, 80
        )
        self.assertEqual(match["route_number"], "21A")

    def test_rejects_uncertain_or_unverified_readings(self):
        self.write_database([
            {"route_number": "21A", "destination": "THIRUMANGALAM"}
        ])

        self.assertIsNone(
            validate_bus.find_verified_bus("21A", "THIRUMANGALAM", 19, 90)
        )
        self.assertIsNone(
            validate_bus.find_verified_bus("21A", "THIRUMANGALAM", 90, 49)
        )
        self.assertIsNone(
            validate_bus.find_verified_bus("21AB", "THIRUMANGALAM", 90, 90)
        )
        self.assertIsNone(
            validate_bus.find_verified_bus("21A", "UNKNOWN", 90, 90)
        )

    def test_legacy_boolean_validator_remains_compatible(self):
        self.write_database([
            {"route_number": "21A", "destination": "THIRUMANGALAM"}
        ])

        self.assertTrue(validate_bus.validate_bus("21a", "thirumangalam"))
        self.assertFalse(validate_bus.validate_bus("21B", "THIRUMANGALAM"))

    def test_rejects_malformed_database_entries(self):
        self.write_database([{"route_number": "21A"}])
        with self.assertRaises(ValueError):
            validate_bus.load_bus_database()

    def test_rejects_tied_route_matches(self):
        self.write_database([
            {"route_number": "21A", "destination": "THIRUMANGALAM"},
            {"route_number": "21AL", "destination": "THIRUMANGALAM"},
        ])

        self.assertIsNone(
            validate_bus.find_verified_bus("21AL", "THIRUMANGALAM", 90, 90)
        )


class MovementDirectionTests(unittest.TestCase):
    def test_area_history_classifies_both_directions(self):
        self.assertEqual(
            _classify_area_trend([100, 102, 104, 106, 108]), "APPROACHING"
        )
        self.assertEqual(
            _classify_area_trend([108, 106, 104, 102, 100]), "LEAVING"
        )

    def test_area_history_reports_uncertain_or_still_collecting(self):
        self.assertEqual(
            _classify_area_trend([100] * MOTION_HISTORY_SIZE),
            "MOVEMENT UNCERTAIN",
        )
        self.assertEqual(_classify_area_trend([100, 101]), "DETECTING DIRECTION")

    def test_sign_read_does_not_depend_on_direction(self):
        self.assertTrue(_should_read_sign(True, False, 1.0, 0.0))
        self.assertFalse(_should_read_sign(False, False, 1.0, 0.0))
        self.assertFalse(_should_read_sign(True, True, 1.0, 0.0))

    def test_bus_verification_uses_two_matching_reads_in_three(self):
        recent_matches = deque(maxlen=3)
        bus = {"route_number": "21A", "destination": "THIRUMANGALAM"}

        self.assertIsNone(_record_bus_candidate(recent_matches, bus))
        self.assertIsNone(_record_bus_candidate(recent_matches, None))
        self.assertEqual(_record_bus_candidate(recent_matches, bus), bus)

    def test_display_separates_bus_number_and_route(self):
        verified_bus = {
            "route_number": "21A",
            "origin": "MADURAI",
            "destination": "THIRUMANGALAM",
        }
        with patch("main.cv2.putText") as draw_text:
            _draw_bus_status(None, "APPROACHING", True, verified_bus)

        displayed_text = [call.args[1] for call in draw_text.call_args_list]
        self.assertIn("Bus number: 21A", displayed_text)
        self.assertIn("Route: MADURAI to THIRUMANGALAM", displayed_text)

    def test_unverified_display_shows_latest_ocr_and_confidence(self):
        reading = {
            "route_text": "21AL!",
            "route_confidence": 30,
            "destination_text": "MADURAI > THIRUMANGALAN",
            "destination_confidence": 80,
        }
        with patch("main.cv2.putText") as draw_text:
            _draw_bus_status(None, "APPROACHING", True, None, reading)

        displayed_text = [call.args[1] for call in draw_text.call_args_list]
        self.assertTrue(any("21AL!" in text and "unverified" in text for text in displayed_text))
        self.assertTrue(any("THIRUMANGALAN" in text and "unverified" in text for text in displayed_text))

    def test_failed_fallback_does_not_replace_primary_route_reading(self):
        primary_reading = {
            "route_text": "21A",
            "route_confidence": 35,
            "destination_text": "MADURAI THIRUMANGALAN",
            "destination_confidence": 30,
        }
        fallback_reading = {
            "route_text": "WRONG",
            "route_confidence": 90,
            "destination_text": "UNRELATED WORDS",
            "destination_confidence": 90,
        }
        with patch("main.read_bus_sign", side_effect=[primary_reading, fallback_reading]):
            with patch("main.find_verified_bus", return_value=None):
                selected_reading, verified_bus = _read_and_verify_sign(
                    object(), (0, 0, 10, 10)
                )

        self.assertIsNone(verified_bus)
        self.assertEqual(selected_reading, primary_reading)


TESSERACT_COMMAND = os.environ.get("TESSERACT_CMD", "")
DEFAULT_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSERACT_AVAILABLE = bool(
    (TESSERACT_COMMAND and Path(TESSERACT_COMMAND).is_file())
    or DEFAULT_TESSERACT.is_file()
    or shutil.which("tesseract")
)


@unittest.skipUnless(TESSERACT_AVAILABLE, "Tesseract executable is not installed")
class SampleOcrTests(unittest.TestCase):
    def test_sample_sign_is_read_and_verified(self):
        import cv2

        from route_ocr_test import read_bus_sign

        image_path = Path(__file__).resolve().parents[1] / "bus.png"
        image = cv2.imread(str(image_path))
        self.assertIsNotNone(image, f"Could not load {image_path}")

        reading = read_bus_sign(image)
        match = validate_bus.find_verified_bus(
            reading["route_text"],
            reading["destination_text"],
            reading["route_confidence"],
            reading["destination_confidence"],
        )
        self.assertIsNotNone(match)
        self.assertEqual(match["route_number"], "21A")
        self.assertIn("MADURAI", reading["destination_text"])

    def test_padded_bus_region_keeps_destination_verifiable(self):
        import cv2

        from route_ocr_test import read_bus_sign

        image_path = Path(__file__).resolve().parents[1] / "bus.png"
        image = cv2.imread(str(image_path))
        reading = read_bus_sign(image, region=(77, 1, 1338, 1073))
        match = validate_bus.find_verified_bus(
            reading["route_text"],
            reading["destination_text"],
            reading["route_confidence"],
            reading["destination_confidence"],
        )

        self.assertIsNotNone(match)
        self.assertEqual(match["route_number"], "21A")

    def test_lower_fallback_band_recovers_sign_inside_bus_region(self):
        import cv2

        from main import _read_and_verify_sign

        image_path = Path(__file__).resolve().parents[1] / "bus.png"
        image = cv2.imread(str(image_path))
        height = image.shape[0]
        sign_band = image[int(height * 0.06):int(height * 0.20), :, :].copy()
        shifted_image = image.copy()
        shifted_image[int(height * 0.06):int(height * 0.20), :, :] = 0
        shifted_top = int(height * 0.28)
        shifted_image[
            shifted_top:shifted_top + sign_band.shape[0], :, :
        ] = sign_band

        reading, match = _read_and_verify_sign(
            shifted_image, (77, 1, 1338, 1073)
        )

        self.assertIsNotNone(match)
        self.assertEqual(match["route_number"], "21A")
        self.assertTrue(reading["route_text"].startswith("21A"))
        self.assertIn("MADURAI", reading["destination_text"])


if __name__ == "__main__":
    unittest.main()