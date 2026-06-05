// Fire-risk color scale: pale ember -> deep red. Used for the map + UI accents.

type Stop = { at: number; rgb: [number, number, number] };

const SCALE: Stop[] = [
  { at: 0.0, rgb: [255, 247, 237] }, // #FFF7ED pale ember
  { at: 0.2, rgb: [254, 215, 170] }, // #FED7AA
  { at: 0.4, rgb: [253, 186, 116] }, // #FDBA74
  { at: 0.55, rgb: [251, 146, 60] }, // #FB923C
  { at: 0.7, rgb: [249, 115, 22] }, // #F97316
  { at: 0.82, rgb: [234, 88, 12] }, // #EA580C
  { at: 1.0, rgb: [185, 28, 28] }, // #B91C1C deep red
];

const NEUTRAL = "#d9d4cf";

function lerp(a: number, b: number, t: number): number {
  return Math.round(a + (b - a) * t);
}

/** Map a 0..1 risk score to a fire-scale hex color. `null` -> neutral gray. */
export function riskColor(score: number | null | undefined): string {
  if (score === null || score === undefined || Number.isNaN(score)) return NEUTRAL;
  const s = Math.min(1, Math.max(0, score));
  let lo = SCALE[0]!;
  let hi = SCALE[SCALE.length - 1]!;
  for (let i = 0; i < SCALE.length - 1; i++) {
    if (s >= SCALE[i]!.at && s <= SCALE[i + 1]!.at) {
      lo = SCALE[i]!;
      hi = SCALE[i + 1]!;
      break;
    }
  }
  const span = hi.at - lo.at || 1;
  const t = (s - lo.at) / span;
  const r = lerp(lo.rgb[0], hi.rgb[0], t);
  const g = lerp(lo.rgb[1], hi.rgb[1], t);
  const b = lerp(lo.rgb[2], hi.rgb[2], t);
  return `rgb(${r}, ${g}, ${b})`;
}

export type RiskBand = "Low" | "Moderate" | "High" | "Extreme" | "No data";

export function riskBand(score: number | null | undefined): RiskBand {
  if (score === null || score === undefined || Number.isNaN(score)) return "No data";
  if (score < 0.35) return "Low";
  if (score < 0.6) return "Moderate";
  if (score < 0.8) return "High";
  return "Extreme";
}

export function formatScore(score: number | null | undefined): string {
  if (score === null || score === undefined || Number.isNaN(score)) return "—";
  return Math.round(score * 100).toString();
}
