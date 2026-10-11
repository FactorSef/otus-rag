# Incident report: Z crash on "Mamont", 2 August 2025

**Printer:** "Mamont" (Tronxy X5S, Marlin 2.1)
**Time:** 2 August 2025, 14:05 (day shift)
**Operator:** Ildar Gabdullin
**Investigated by:** Varvara Tikhomirova
**Damage:** PEI sheet scratched across 180 mm, nozzle bent; 5 200 RUB in parts. No injuries.

## What happened

Ildar started job #4317 (a 290 mm PETG drone frame) on "Mamont". During homing, the Z axis did not stop at the endstop: the nozzle drove into the bed, scraped across the PEI sheet while the X/Y homing continued, and bent the nozzle. Ildar hit the emergency stop after about four seconds.

## Root cause

The **mechanical Z endstop cable was broken inside its insulation**, right where it bends at the gantry. In 2023 the switch had been rewired as normally-open. With a normally-open switch, a broken wire looks exactly like "endstop not reached", so Marlin kept lowering Z.

Contributing factors:

- The endstop cable ran without any strain relief along the moving gantry.
- The pre-print checklist for "Mamont" did not include an endstop test (`M119`).

## Actions

| Action | Owner | Due | Status |
|---|---|---|---|
| Replace the mechanical Z endstop with an **optical endstop** wired normally-closed, so a broken wire reads as "triggered" | Varvara Tikhomirova | 2025-08-09 | done |
| Add a printed strain relief and route the endstop cable through the cable chain | Ildar Gabdullin | 2025-08-09 | done |
| Add `M119` endstop check to the morning routine for Marlin printers ("Soyka", "Mamont") | Ildar Gabdullin | 2025-08-05 | done |
| Replace PEI sheet and nozzle | Ildar Gabdullin | 2025-08-04 | done |

## Lesson

Wire every endstop **normally-closed**: then a broken wire stops the axis instead of crashing it. This rule now applies to all future printers (see `procedures/printer_onboarding.md`).
