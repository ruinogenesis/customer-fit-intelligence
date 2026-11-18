// The love matrix's geometry and counts: plain functions, with no page and no
// Firebase, so node --test can check them (dashboard/tests).

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

/** The quadrants as the API stores them, with the names people read. */
export const QUADRANTS = Object.freeze({
  lovemark: "Lovemark",
  brand: "Brand",
  fad: "Fad",
  product: "Product",
});

/** A score from 0 to 1 as a distance along an axis, in SVG units. */
export function along(score) {
  return MARGIN + score * (SIZE - 2 * MARGIN);
}

/**
 * One customer document as a dot, or null while it has no scores.
 * Respect runs left to right, Love bottom to top.
 */
export function toDot(id, customer) {
  const scores = customer.scores;
  const placed =
    scores != null &&
    Number.isFinite(scores.respect) &&
    Number.isFinite(scores.love) &&
    Object.hasOwn(QUADRANTS, customer.quadrant);
  if (!placed) {
    return null;
  }
  return {
    id,
    x: along(scores.respect),
    y: SIZE - along(scores.love),
    quadrant: customer.quadrant,
  };
}

/** How many dots each quadrant holds, with every quadrant listed. */
export function countByQuadrant(dots) {
  const counts = Object.fromEntries(Object.keys(QUADRANTS).map((q) => [q, 0]));
  for (const dot of dots) {
    counts[dot.quadrant] += 1;
  }
  return counts;
}
