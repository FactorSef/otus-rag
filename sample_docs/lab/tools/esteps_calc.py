"""Extruder calibration calculator used during printer onboarding.

Supports both firmware families used in the lab:

* Marlin printers ("Soyka", "Mamont") are calibrated in **steps per mm** (M92 E...).
* Klipper printers ("Kedr", "Zholud") are calibrated with **rotation_distance**.

The procedure is the one from the wiki page "Printer calibration": mark 120 mm of
filament, extrude 100 mm, measure what is left to the mark.

Example (the 2025-06-14 calibration of "Soyka"):
    python esteps_calc.py marlin --current 93.0 --remaining 24.42
    -> new e-steps: 97.3
"""

import argparse

MARK_MM = 120.0
REQUESTED_MM = 100.0


def actual_extruded(remaining_to_mark: float) -> float:
    """Length actually extruded, given the distance left from the extruder to the mark."""
    return MARK_MM - remaining_to_mark


def new_esteps(current_esteps: float, remaining_to_mark: float) -> float:
    """Marlin: steps_new = steps_old * requested / actual."""
    return round(current_esteps * REQUESTED_MM / actual_extruded(remaining_to_mark), 1)


def new_rotation_distance(current_rd: float, remaining_to_mark: float) -> float:
    """Klipper: rotation_distance_new = rotation_distance_old * actual / requested."""
    return round(current_rd * actual_extruded(remaining_to_mark) / REQUESTED_MM, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rusty Goose extruder calibration")
    sub = parser.add_subparsers(dest="firmware", required=True)
    m = sub.add_parser("marlin")
    m.add_argument("--current", type=float, required=True, help="current e-steps/mm")
    m.add_argument("--remaining", type=float, required=True, help="mm left to the 120 mm mark")
    k = sub.add_parser("klipper")
    k.add_argument("--current", type=float, required=True, help="current rotation_distance")
    k.add_argument("--remaining", type=float, required=True, help="mm left to the 120 mm mark")
    args = parser.parse_args()

    extruded = actual_extruded(args.remaining)
    print(f"extruded {extruded:.2f} mm of {REQUESTED_MM:.0f} mm requested")
    if args.firmware == "marlin":
        value = new_esteps(args.current, args.remaining)
        print(f"new e-steps: {value}  ->  send M92 E{value} and M500")
    else:
        value = new_rotation_distance(args.current, args.remaining)
        print(f"new rotation_distance: {value}  ->  update [extruder] in printer.cfg")


if __name__ == "__main__":
    main()
