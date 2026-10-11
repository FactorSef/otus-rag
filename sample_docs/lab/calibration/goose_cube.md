# The Goose Cube — internal calibration standard

The Goose Cube is the Rusty Goose lab's own calibration print. It replaces the classic 20 mm calibration cube because it checks more things in one print and is harder to "pass by accident".

## Geometry

- Outer size: **23.7 × 23.7 × 23.7 mm** (the odd size is deliberate: a printer with wrong steps per mm cannot match it by rounding luck).
- A **goose-head silhouette** embossed 0.6 mm on the front face — checks small detail and retraction.
- A **through hole of 6.35 mm** (1/4 inch) along the Z axis — checks hole shrinkage.
- A **45° overhang notch** on the right face — checks part cooling.
- The letter of the printer's name embossed on top (С for Soyka, К for Kedr, etc.), so cubes from different printers never get mixed up.

## Tolerance

All three outer dimensions must be within **±0.08 mm** of 23.7 mm, measured with a digital caliper after the cube has cooled for at least 10 minutes. The hole must measure between 6.20 and 6.40 mm.

| Measurement | Nominal | Accept range |
|---|---|---|
| X | 23.70 mm | 23.62–23.78 mm |
| Y | 23.70 mm | 23.62–23.78 mm |
| Z | 23.70 mm | 23.62–23.78 mm |
| Hole Ø | 6.35 mm | 6.20–6.40 mm |

## Schedule

- Every printer prints one Goose Cube **every Monday morning** in its default material.
- After any nozzle change, extruder rebuild or firmware update — immediately.
- New printers print three cubes during onboarding.

Results are written into the calibration sheet with the date, printer name and the three dimensions. Two failed cubes in a row take the printer out of production until Varvara Tikhomirova has checked it.

## Typical failures and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| X and Y both too small by the same amount | under-extrusion or wrong flow | recalibrate e-steps / rotation_distance, then flow |
| Only one axis off | belt tension or wrong steps for that axis | check the belt (see wiki page on belts), then steps |
| Z off by a multiple of the layer height | Z steps or first layer squish | check Z steps, re-do Z offset |
| Hole too small | normal plastic shrinkage | apply hole compensation in the slicer |
| Goose head blobby | too much pressure / retraction issues | tune pressure advance (Klipper) or linear advance (Marlin) |
