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

/**
 * Which customers the list shows: a quadrant, a least churn risk, or both;
 * or the customers on one spot, picked by keyboard (T409).
 */
export interface Selection {
  quadrant: Quadrant | null;
  minChurn: number;
  ids?: readonly string[] | null;
}

export function matches(dot: Dot, { quadrant, minChurn, ids }: Selection): boolean {
  if (ids) return ids.includes(dot.id);
  return (quadrant === null || dot.quadrant === quadrant) && dot.churn >= minChurn;
}

// --- Moving through the dots by keyboard (T409) ---

/** The customers drawn at one exact place: one, or several whose scores match. */
export interface Spot {
  key: string;
  x: number;
  y: number;
  /** By id. */
  dots: Dot[];
}

/** Every spot, in reading order: top to bottom, then left to right. */
export function spotsOf(dots: readonly Dot[]): Spot[] {
  const spots = new Map<string, Spot>();
  for (const dot of dots) {
    const key = `${dot.x},${dot.y}`;
    const spot = spots.get(key) ?? { key, x: dot.x, y: dot.y, dots: [] };
    spot.dots.push(dot);
    spots.set(key, spot);
  }
  for (const spot of spots.values()) spot.dots.sort((a, b) => a.id.localeCompare(b.id));
  return [...spots.values()].sort((a, b) => a.y - b.y || a.x - b.x);
}

export const ARROWS = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"] as const;
export type Arrow = (typeof ARROWS)[number];

/**
 * The spot an arrow key moves to: the nearest within 45 degrees of the
 * arrow's direction, sideways distance counting double; null at the edge.
 * On dev-like data the arrows reach all 175 spots, about 11 presses apart on
 * average; but on some layouts a spot is never anyone's nearest (T409 found
 * 20 of 300 random layouts leaving up to 5% unreached). Page Up and Page
 * Down step through every spot in reading order, so none is ever out of
 * reach.
 */
export function nextSpot(spots: readonly Spot[], from: Spot, arrow: Arrow): Spot | null {
  let best: { score: number; spot: Spot } | null = null;
  for (const spot of spots) {
    const dx = spot.x - from.x;
    const dy = spot.y - from.y;
    const [ahead, aside] = {
      ArrowRight: [dx, dy],
      ArrowLeft: [-dx, dy],
      ArrowDown: [dy, dx],
      ArrowUp: [-dy, dx],
    }[arrow];
    if (ahead <= 0 || Math.abs(aside) > ahead) continue;
    const score = ahead + 2 * Math.abs(aside);
    if (best === null || score < best.score) best = { score, spot };
  }
  return best?.spot ?? null;
}

/** The spot nearest a place: where focus goes when its spot moves away. */
export function nearestSpot(spots: readonly Spot[], x: number, y: number): Spot | null {
  let best: Spot | null = null;
  for (const spot of spots) {
    if (best === null || Math.hypot(spot.x - x, spot.y - y) < Math.hypot(best.x - x, best.y - y)) {
      best = spot;
    }
  }
  return best;
}

/** A place on the plot as scores: Respect across, Love up. */
export function scoresAt(x: number, y: number): { respect: number; love: number } {
  const side = SIZE - 2 * MARGIN;
  return { respect: (x - MARGIN) / side, love: (SIZE - y - MARGIN) / side };
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
