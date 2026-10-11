// Rusty Goose lab — order price calculator (used by the order form on the lab's tablet).
//
// Price = max(MINIMUM_ORDER_RUB, weight * PRICE_PER_GRAM_RUB * materialFactor) * rushFactor
//
// Prices were last reviewed at the planning meeting on 2025-09-29: the proposed raise
// to 4.60 RUB/g was postponed to December.

'use strict';

const PRICE_PER_GRAM_RUB = 4.35;
const MINIMUM_ORDER_RUB = 740;
const RUSH_MULTIPLIER = 1.85; // order ready within 24 hours
const MULTICOLOR_SURCHARGE_RUB = 250; // per extra colour on "Pustelga" / "Klyukva"

// Relative to PLA/PETG. Engineering materials cost more and print slower.
const MATERIAL_FACTORS = {
  PLA: 1.0,
  PETG: 1.0,
  'SMOKY_QUARTZ_PETG': 1.2, // house material, dried before every print
  TPU: 1.6,
  ABS: 1.3,
  ASA: 1.4,
  PVA: 2.2,
  'PA12-CF': 3.1,
};

function materialFactor(material) {
  const key = String(material).toUpperCase().replace(/\s+/g, '_');
  if (!(key in MATERIAL_FACTORS)) {
    throw new Error(`Unknown material "${material}". Known: ${Object.keys(MATERIAL_FACTORS).join(', ')}`);
  }
  return MATERIAL_FACTORS[key];
}

/**
 * @param {object} order
 * @param {number} order.grams      weight of the part including supports, from the slicer
 * @param {string} order.material   key of MATERIAL_FACTORS
 * @param {number} [order.colors=1] number of colours
 * @param {boolean} [order.rush=false]
 * @returns {{total: number, breakdown: object}}
 */
function quote({ grams, material, colors = 1, rush = false }) {
  if (!(grams > 0)) throw new Error('grams must be positive');
  const base = grams * PRICE_PER_GRAM_RUB * materialFactor(material);
  const withColors = base + Math.max(0, colors - 1) * MULTICOLOR_SURCHARGE_RUB;
  const withMinimum = Math.max(MINIMUM_ORDER_RUB, withColors);
  const total = Math.round(withMinimum * (rush ? RUSH_MULTIPLIER : 1));
  return {
    total,
    breakdown: {
      base: Math.round(base),
      colors: Math.max(0, colors - 1) * MULTICOLOR_SURCHARGE_RUB,
      minimumApplied: withColors < MINIMUM_ORDER_RUB,
      rush,
    },
  };
}

module.exports = {
  quote,
  PRICE_PER_GRAM_RUB,
  MINIMUM_ORDER_RUB,
  RUSH_MULTIPLIER,
  MULTICOLOR_SURCHARGE_RUB,
  MATERIAL_FACTORS,
};

// Quick self-check: node pricing.js
if (require.main === module) {
  console.log(quote({ grams: 12, material: 'PLA' }));                 // minimum 740 applies
  console.log(quote({ grams: 380, material: 'Smoky Quartz PETG' }));  // 1984
  console.log(quote({ grams: 120, material: 'TPU', rush: true }));    // 1545
}
