# Night Shift Policy (checklist NS-7)

**Owner:** Varvara Tikhomirova (maintenance lead)
**Applies to:** night operator (currently Timofey Lastochkin) and anyone starting a print after 21:00
**Revision:** 4, effective 24 March 2025 (revised after the "blue cloud" incident of 17 March 2025)

## 1. Why this policy exists

The Rusty Goose lab runs prints around the clock. At night the building is empty for long stretches, so every unattended print is a fire risk. After the 17 March 2025 thermal runaway on the printer "Kedr" (see `incidents/2025-03-17-blue-cloud.md`), the rules below became mandatory.

## 2. Hard limits

- **Maximum unattended print duration: 9 h 40 min.** Any job whose slicer estimate exceeds this must either be split into parts or be scheduled for the day shift, when someone is in room B-14.
- Only printers with **firmware thermal runaway protection** may print unattended: in Klipper this means `verify_heater` is active for both the extruder and the bed; in Marlin it means `THERMAL_PROTECTION_HOTENDS` and `THERMAL_PROTECTION_BED` are enabled.
- The printer **"Soyka"** (Anet A8) never prints unattended at night, regardless of firmware. It is the oldest machine in the fleet and still has the original acrylic frame.
- ABS and ASA may be printed at night **only on "Kedr"**, and only with the chamber door closed and the chamber exhaust filter running.

## 3. Checklist NS-7

The night operator walks through all seven items before leaving the room. Each item is ticked in the paper log on the door of room B-14.

1. **Camera** — the printer's camera stream opens on the night operator's phone and shows the nozzle.
2. **First layer** — the first layer has finished and looks correct (no lifted corners, no gaps).
3. **Spool** — there is enough filament for the whole job plus 15 % (weigh the spool if unsure).
4. **Dry filament** — the filament came out of the dry room B-16 less than 8 hours ago, or out of a sealed dry box.
5. **Clear surroundings** — nothing flammable within 50 cm of the printer: no paper, no cardboard boxes, no isopropyl alcohol bottles.
6. **Extinguisher** — the powder extinguisher OP-4 in room B-14 is in place and its pressure gauge is in the green zone. The automatic aerosol extinguisher above the rack has a green status LED.
7. **Smoke detector** — the smoke detector test button was pressed and the alarm reached the night operator's phone.

If any item fails, the print is cancelled or postponed. No exceptions.

## 4. Remote monitoring

- The night operator checks every running printer through its camera **at least once every 90 minutes**.
- If the camera shows spaghetti, detached parts or smoke, the operator stops the printer remotely (`CANCEL_PRINT` in Klipper / "Stop" in the Bambu app), and then cuts power with the smart socket.
- Any smoke or burning smell is treated as a fire until proven otherwise: call the fire service first, then Varvara.

## 5. Morning handover

At 08:30 the night operator hands over to the day operator (currently Ildar Gabdullin):

- list of finished, failed and still running jobs (job numbers from the print log);
- any warnings shown by the printers (thermal, filament runout, toolhead errors);
- filament consumed during the night, so the stock sheet stays correct.

## 6. Change history

| Rev. | Date | Change |
|---|---|---|
| 1 | 2024-02-12 | First version, written when the night shift started |
| 2 | 2024-06-03 | Added camera check every 90 minutes |
| 3 | 2024-11-18 | ABS/ASA at night restricted to "Kedr" |
| 4 | 2025-03-24 | Max unattended duration 9 h 40 min; checklist renamed NS-7 (was NS-5); "Soyka" banned from unattended night prints |
