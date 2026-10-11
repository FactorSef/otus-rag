// Rusty Goose lab — print job queue helper.
//
// Assigns incoming jobs to printers according to the material rules from fleet.yaml
// and the night shift policy (checklist NS-7).

'use strict';

const MAX_UNATTENDED_MINUTES = 9 * 60 + 40; // 9 h 40 min, night shift policy rev. 4
const NIGHT_START_HOUR = 21;
const NIGHT_END_HOUR = 8;

// Which printer may print which material. Order = preference.
const MATERIAL_ROUTING = {
  PLA: ['pustelga', 'klyukva', 'soyka', 'kedr'],
  PETG: ['mamont', 'pustelga', 'kedr', 'klyukva'],
  'SMOKY_QUARTZ_PETG': ['mamont', 'pustelga', 'kedr', 'klyukva'],
  TPU: ['zholud', 'klyukva'],
  ABS: ['kedr'],
  ASA: ['kedr'],
  PVA: ['klyukva'],
  'PA12-CF': ['kedr'],
};

// Printers that never run unattended at night.
const NO_NIGHT_PRINTING = new Set(['soyka']);

// Largest single part per printer, mm (X, Y, Z).
const BUILD_VOLUMES = {
  soyka: [220, 220, 240],
  kedr: [350, 350, 330],
  pustelga: [256, 256, 256],
  zholud: [220, 220, 270],
  klyukva: [270, 270, 270],
  mamont: [330, 330, 400],
};

function isNight(date) {
  const h = date.getHours();
  return h >= NIGHT_START_HOUR || h < NIGHT_END_HOUR;
}

function fits(size, volume) {
  const a = [...size].sort((x, y) => x - y);
  const b = [...volume].sort((x, y) => x - y);
  return a.every((v, i) => v <= b[i]);
}

/**
 * Pick a printer for a job.
 * @param {{id: number, material: string, sizeMm: number[], estimateMinutes: number, colors?: number}} job
 * @param {Date} startAt
 * @param {Set<string>} busy  printers currently printing
 */
function assign(job, startAt, busy = new Set()) {
  const key = job.material.toUpperCase().replace(/\s+/g, '_');
  const candidates = MATERIAL_ROUTING[key];
  if (!candidates) return { ok: false, reason: `material ${job.material} is not printed in the lab` };

  const night = isNight(startAt);
  if (night && job.estimateMinutes > MAX_UNATTENDED_MINUTES) {
    return { ok: false, reason: `job #${job.id} takes ${job.estimateMinutes} min, longer than the 9 h 40 min night limit: split it or start it in the day shift` };
  }
  for (const printer of candidates) {
    if (busy.has(printer)) continue;
    if (night && NO_NIGHT_PRINTING.has(printer)) continue;
    if (!fits(job.sizeMm, BUILD_VOLUMES[printer])) continue;
    if ((job.colors ?? 1) > 1 && !['pustelga', 'klyukva'].includes(printer)) continue;
    return { ok: true, printer };
  }
  return { ok: false, reason: `no free printer for job #${job.id} (${job.material}, ${job.sizeMm.join('x')} mm)` };
}

module.exports = { assign, MAX_UNATTENDED_MINUTES, MATERIAL_ROUTING, BUILD_VOLUMES };
