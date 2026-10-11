// Rusty Goose lab — preventive maintenance scheduler.
//
// Reads printing hours per printer (exported from the print log) and tells the day
// operator what is due. Intervals come from fleet.yaml; these are the defaults.

export const NOZZLE_SWAP_INTERVAL_HOURS = 312;
export const BELT_CHECK_INTERVAL_HOURS = 90;
export const RAIL_LUBE_INTERVAL_HOURS = 500;
export const GOOSE_CUBE_WEEKDAY = 1; // Monday: weekly calibration print (calibration/goose_cube.md)

/** Warn this many hours before a task becomes due. */
export const EARLY_WARNING_HOURS = 20;

export type PrinterName = 'soyka' | 'kedr' | 'pustelga' | 'zholud' | 'klyukva' | 'mamont';

export interface PrinterHours {
  printer: PrinterName;
  sinceNozzleSwap: number;
  sinceBeltCheck: number;
  sinceRailLube?: number; // only printers with linear rails (Kedr)
}

export interface Overrides {
  nozzleSwap?: number;
  beltCheck?: number;
  railLube?: number;
}

/** Per-printer interval overrides (keep in sync with fleet.yaml). */
export const OVERRIDES: Partial<Record<PrinterName, Overrides>> = {
  // Reduced at the 2025-09-29 planning meeting: Mamont prints large, heavy parts.
  mamont: { beltCheck: 75 },
};

export type Status = 'ok' | 'soon' | 'due';

export interface Task {
  printer: PrinterName;
  task: 'nozzle swap' | 'belt check' | 'rail lubrication';
  hours: number;
  interval: number;
  status: Status;
}

function status(hours: number, interval: number): Status {
  if (hours >= interval) return 'due';
  if (hours >= interval - EARLY_WARNING_HOURS) return 'soon';
  return 'ok';
}

export function tasksFor(h: PrinterHours): Task[] {
  const o = OVERRIDES[h.printer] ?? {};
  const nozzle = o.nozzleSwap ?? NOZZLE_SWAP_INTERVAL_HOURS;
  const belt = o.beltCheck ?? BELT_CHECK_INTERVAL_HOURS;
  const lube = o.railLube ?? RAIL_LUBE_INTERVAL_HOURS;

  const tasks: Task[] = [
    { printer: h.printer, task: 'nozzle swap', hours: h.sinceNozzleSwap, interval: nozzle, status: status(h.sinceNozzleSwap, nozzle) },
    { printer: h.printer, task: 'belt check', hours: h.sinceBeltCheck, interval: belt, status: status(h.sinceBeltCheck, belt) },
  ];
  if (h.sinceRailLube !== undefined) {
    tasks.push({ printer: h.printer, task: 'rail lubrication', hours: h.sinceRailLube, interval: lube, status: status(h.sinceRailLube, lube) });
  }
  return tasks;
}

export function dueReport(all: PrinterHours[]): string {
  const rows = all
    .flatMap(tasksFor)
    .filter((t) => t.status !== 'ok')
    .sort((a, b) => (a.status === b.status ? b.hours / b.interval - a.hours / a.interval : a.status === 'due' ? -1 : 1));
  if (rows.length === 0) return 'Nothing due. Go feed... no, do NOT feed Shurup.';
  return rows
    .map((t) => `${t.status.toUpperCase().padEnd(4)} ${t.printer.padEnd(9)} ${t.task.padEnd(17)} ${t.hours}/${t.interval} h`)
    .join('\n');
}

// Snapshot from the 2025-09-29 planning meeting.
export const SNAPSHOT_2025_09_29: PrinterHours[] = [
  { printer: 'soyka', sinceNozzleSwap: 121, sinceBeltCheck: 33 },
  { printer: 'kedr', sinceNozzleSwap: 244, sinceBeltCheck: 61, sinceRailLube: 410 },
  { printer: 'pustelga', sinceNozzleSwap: 187, sinceBeltCheck: 88 },
  { printer: 'zholud', sinceNozzleSwap: 298, sinceBeltCheck: 52 },
  { printer: 'klyukva', sinceNozzleSwap: 96, sinceBeltCheck: 40 },
  { printer: 'mamont', sinceNozzleSwap: 150, sinceBeltCheck: 71 },
];
