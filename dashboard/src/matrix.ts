// The love matrix's geometry, dots and counts (T206, brought into the app by
// T403): plain functions, with no page and no Firebase.
import { QUADRANTS, type Quadrant, type QuadrantCounts } from "./overview";

/** The plot's width and height, in SVG units. */
export const SIZE = 640;

/** Room around the plot for the axis labels, in SVG units. */
export const MARGIN = 56;

/**
 * Where D5 draws both quadrant lines. A customer's quadrant comes from their
 * own document, which the model's lines decided; this is only where the
 * lines are drawn.
 */
export const LINE = 0.5;

/** At or above this churn risk a customer is at risk, as on the Overview (T402). */
export const AT_RISK = 0.1;

/** A customer as the live feed gives them: what the matrix needs. */
export interface FeedCustomer {
  customer_id: string;
  scores: { love: number; respect: number; fit: number } | null;
  churn_risk: number | null;
  quadrant: Quadrant | null;
}

export interface Dot {
  id: string;
  x: number;
  y: number;
  quadrant: Quadrant;
  churn: number;
  atRisk: boolean;
}

/** A score from 0 to 1 as a distance along an axis, in SVG units. */
export function along(score: number): number {
  return MARGIN + score * (SIZE - 2 * MARGIN);
}

function isQuadrant(value: unknown): value is Quadrant {
  return typeof value === "string" && (QUADRANTS as readonly string[]).includes(value);
}

/**
 * A customer as a dot, or null while they have no scores. Respect runs left
 * to right, Love bottom to top.
 */
export function toDot(customer: FeedCustomer): Dot | null {
  const { scores, quadrant, churn_risk: churn } = customer;
  const placed =
    scores !== null &&
    Number.isFinite(scores.respect) &&
    Number.isFinite(scores.love) &&
    typeof churn === "number" &&
    isQuadrant(quadrant);
  if (!placed) return null;
  return {
    id: customer.customer_id,
    // To a hundredth of a unit: no float tails in the page.
    x: Math.round(along(scores.respect) * 100) / 100,
    y: Math.round((SIZE - along(scores.love)) * 100) / 100,
    quadrant,
    churn,
    atRisk: churn >= AT_RISK,
  };
}

/** How many dots each quadrant holds, with every quadrant listed. */
export function countByQuadrant(dots: readonly Dot[]): QuadrantCounts {
  const counts = Object.fromEntries(QUADRANTS.map((q) => [q, 0])) as QuadrantCounts;
  for (const dot of dots) counts[dot.quadrant] += 1;
  return counts;
}

/** Which customers the list shows: a quadrant, a least churn risk, or both. */
export interface Selection {
  quadrant: Quadrant | null;
  minChurn: number;
}

export function matches(dot: Dot, { quadrant, minChurn }: Selection): boolean {
  return (quadrant === null || dot.quadrant === quadrant) && dot.churn >= minChurn;
}

export const NOT_SCORED = "unscored";

/** GET /matrix with from and to: by quadrant on from (or not scored), then on to. */
export interface Movement {
  from: string;
  to: string;
  moves: Record<Quadrant | typeof NOT_SCORED, QuadrantCounts>;
  customers: number;
}

/** How many stayed in their quadrant, moved to another, or were first scored. */
export function moveTotals(movement: Movement): { stayed: number; moved: number; scored: number } {
  let stayed = 0;
  let moved = 0;
  for (const was of QUADRANTS) {
    for (const now of QUADRANTS) {
      const n = movement.moves[was][now];
      if (was === now) stayed += n;
      else moved += n;
    }
  }
  const scored = QUADRANTS.reduce((sum, now) => sum + movement.moves[NOT_SCORED][now], 0);
  return { stayed, moved, scored };
}

/** A Bangkok calendar day, days before the given moment's: 2026-11-17. */
export function bangkokDay(at: Date, daysBefore = 0): string {
  const today = new Intl.DateTimeFormat("sv-SE", { timeZone: "Asia/Bangkok" }).format(at);
  const day = new Date(`${today}T00:00:00Z`);
  day.setUTCDate(day.getUTCDate() - daysBefore);
  return day.toISOString().slice(0, 10);
}

/** A day, n days later (or earlier, for a negative n): 2026-11-12 for -7. */
export function addDays(day: string, n: number): string {
  const d = new Date(`${day}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + n);
  return d.toISOString().slice(0, 10);
}
