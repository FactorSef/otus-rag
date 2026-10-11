"""Standard Rusty Goose purge line generator.

Every printer in the lab starts a print with the same purge line, so operators can
judge extrusion at a glance: a line that is too thin means under-extrusion or a
partial clog, a wavy line means the first layer is too low.

Usage:
    python purge.py --printer kedr --temp 255 > purge.gcode
"""

import argparse
import sys

# Fixed for all printers (procedure "Printer Onboarding", step 7).
PURGE_LINE_LENGTH_MM = 143
PURGE_LINE_WIDTH_MM = 0.8
PURGE_LAYER_HEIGHT_MM = 0.3
PURGE_FEEDRATE = 1170          # mm/min — slow enough for TPU on "Zholud"
PURGE_START_X = 5.0
PURGE_START_Y = 5.0
FILAMENT_DIAMETER_MM = 1.75
WIPE_DISTANCE_MM = 12

# Printers whose bed is smaller than 150 mm along X after the clips are installed
# draw the line along Y instead.
DRAW_ALONG_Y = {"soyka"}


def filament_length(line_length: float, width: float, height: float) -> float:
    """Filament (mm) needed to extrude a rectangular line of the given size."""
    area_line = width * height
    area_filament = 3.14159265 * (FILAMENT_DIAMETER_MM / 2) ** 2
    return line_length * area_line / area_filament


def purge_gcode(printer: str, nozzle_temp: int) -> str:
    e_len = filament_length(PURGE_LINE_LENGTH_MM, PURGE_LINE_WIDTH_MM, PURGE_LAYER_HEIGHT_MM)
    along_y = printer in DRAW_ALONG_Y
    end_x = PURGE_START_X if along_y else PURGE_START_X + PURGE_LINE_LENGTH_MM
    end_y = PURGE_START_Y + PURGE_LINE_LENGTH_MM if along_y else PURGE_START_Y
    wipe_x = end_x if along_y else end_x + WIPE_DISTANCE_MM
    wipe_y = end_y + WIPE_DISTANCE_MM if along_y else end_y
    lines = [
        f"; Rusty Goose purge line for {printer}",
        f"M109 S{nozzle_temp}            ; wait for nozzle temperature",
        "G92 E0",
        f"G1 Z{PURGE_LAYER_HEIGHT_MM:.2f} F600",
        f"G1 X{PURGE_START_X:.1f} Y{PURGE_START_Y:.1f} F6000",
        f"G1 X{end_x:.1f} Y{end_y:.1f} E{e_len:.3f} F{PURGE_FEEDRATE}",
        f"G1 X{wipe_x:.1f} Y{wipe_y:.1f} F6000  ; wipe without extruding",
        "G92 E0",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--printer", required=True, help="printer name from fleet.yaml")
    parser.add_argument("--temp", type=int, required=True, help="nozzle temperature, °C")
    args = parser.parse_args(argv)
    if not 170 <= args.temp <= 300:
        print("nozzle temperature out of the lab's allowed range 170-300 °C", file=sys.stderr)
        return 2
    sys.stdout.write(purge_gcode(args.printer.lower(), args.temp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
