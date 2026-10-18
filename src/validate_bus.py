
import json
import math
import re
from difflib import SequenceMatcher
from pathlib import Path

DATABASE_FILE = Path(__file__).with_name("bus_database.json")
MIN_ROUTE_CONFIDENCE = 20.0
MIN_DESTINATION_CONFIDENCE = 50.0
MIN_DESTINATION_SIMILARITY = 0.90


def _compact_text(value):
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def load_bus_database():
    with DATABASE_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)
    buses = data.get("buses")
    if not isinstance(buses, list):
        raise ValueError("Bus database must contain a 'buses' list.")
    for bus in buses:
        if (
            not isinstance(bus, dict)
            or not isinstance(bus.get("route_number"), str)
            or not isinstance(bus.get("destination"), str)
            or not bus["route_number"].strip()
            or not bus["destination"].strip()
        ):
            raise ValueError(
                "Each bus entry must have non-empty route_number and destination strings."
            )
    return buses


def find_verified_bus(
    route_number,
    destination,
    route_confidence=100.0,
    destination_confidence=100.0,
):
    try:
        route_confidence = float(route_confidence)
        destination_confidence = float(destination_confidence)
    except (TypeError, ValueError):
        return None
    if (
        not math.isfinite(route_confidence)
        or route_confidence < MIN_ROUTE_CONFIDENCE
        or not math.isfinite(destination_confidence)
        or destination_confidence < MIN_DESTINATION_CONFIDENCE
    ):
        return None

    observed_route = _compact_text(route_number)
    observed_destination = _compact_text(destination)
    if not observed_route or not observed_destination:
        return None

    buses = load_bus_database()
    matching_buses = []
    for bus in buses:
        route_key = _compact_text(bus.get("route_number", ""))
        if not route_key:
            continue
        trailing_confusion = (
            route_key[-1].isalpha()
            and len(observed_route) == len(route_key) + 1
            and observed_route.startswith(route_key)
            and observed_route[-1] in "IL1"
        )
        route_matches = observed_route == route_key or trailing_confusion
        if route_matches:
            expected_destination = _compact_text(bus.get("destination", ""))
            if not expected_destination:
                continue
            observed_suffix = observed_destination[-len(expected_destination):]
            similarity = SequenceMatcher(
                None, observed_suffix, expected_destination
            ).ratio()
            if similarity >= MIN_DESTINATION_SIMILARITY:
                matching_buses.append((similarity, bus))

    if not matching_buses:
        return None

    best_score = max(score for score, _ in matching_buses)
    best_matches = [
        bus for score, bus in matching_buses if score == best_score
    ]
    if len(best_matches) != 1:
        return None
    return best_matches[0]


def validate_bus(route_number, destination):
    return find_verified_bus(route_number, destination) is not None


if __name__ == "__main__":
    route = input("Enter detected route number: ")
    destination = input("Enter detected destination: ")

    if validate_bus(route, destination):
        print("Bus verified successfully!")
    else:
        print("Bus not found in database. Do not announce yet.")
