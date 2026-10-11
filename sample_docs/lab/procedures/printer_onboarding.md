# Printer Onboarding Procedure

How a new printer enters production at the Rusty Goose lab. Written by Varvara Tikhomirova after "Klyukva" (Snapmaker U1) joined the fleet in 2025.

## 1. Naming

Every printer gets a name before it prints its first part. Names come from Siberian birds, trees and berries: Soyka (jay), Kedr (cedar), Pustelga (kestrel), Zholud (acorn), Klyukva (cranberry). The only exception is "Mamont" (mammoth), named for its size.

The name is printed on a label on the frame and used everywhere: in `fleet.yaml`, in the print log, in incident reports and in the job queue.

## 2. Acceptance checklist

A new printer must pass all of these before it gets real orders:

1. **Mechanical inspection** — belts tensioned, frame square, all screws checked with threadlocker where the manufacturer allows it.
2. **Firmware** — latest stable firmware installed. Thermal runaway protection verified by unplugging the thermistor during a heat-up (the printer must shut down the heater within the configured time).
3. **PID tuning** — run PID calibration for the hotend at the most common printing temperature (e.g. `PID_CALIBRATE HEATER=extruder TARGET=235` on Klipper) and for the bed.
4. **Extruder calibration** — measure and set e-steps (Marlin) or `rotation_distance` (Klipper). Our own calculator is `tools/esteps_calc.py`.
5. **Input shaping / pressure advance** — on Klipper printers, measure resonances and tune pressure advance for each filament we plan to use on that printer.
6. **Goose Cube** — print three Goose Cubes (see `calibration/goose_cube.md`). All dimensions must be within ±0.08 mm.
7. **Purge line** — configure the start G-code to draw the standard Rusty Goose purge line. Its length is fixed at 143 mm on all printers (see `tools/purge.py`).
8. **72-hour burn-in** — print continuously for 72 hours (day shift only, never unattended at night) with test parts.

## 3. Registration

After acceptance:

- add the printer to `fleet.yaml` with its firmware, nozzle, calibration values and allowed materials;
- add a camera stream and smart socket to the night monitoring dashboard;
- add maintenance intervals to `tools/maintenance.ts` if they differ from the defaults (nozzle swap every 312 printing hours, belt check every 90 printing hours, rail lubrication every 500 printing hours).

## 4. Special notes per printer type

- **Snapmaker U1 ("Klyukva")** — before printing TPU, reduce *Material Switching Retraction* from the default 10 mm to **3 mm**, otherwise TPU buckles around the extruder gears during a tool swap. Also run the automatic toolhead offset calibration after every nozzle change.
- **Bambu Lab P1P ("Pustelga")** — keep the AMS in room B-14, not in the dry room: the AMS cable is only 1.5 m long. Use the lab desiccant boxes inside the AMS.
- **Voron 2.4 ("Kedr")** — after the 2025 incident the thermistor cartridge is held by a printed polycarbonate clamp in addition to the stock grub screw.
