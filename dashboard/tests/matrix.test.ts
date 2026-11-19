// T206, in the app since T403: the love matrix's geometry, dots and counts;
// moving through its spots by keyboard (T409).
import { expect, test } from "vitest";

import {
  along,
  AT_RISK,
  bangkokDay,
  countByQuadrant,
  LINE,
  MARGIN,
  matches,
  moveTotals,
  nearestSpot,
  nextSpot,
  scoresAt,
  SIZE,
  spotsOf,
  lovesAsExpected,
  toDot,
  type Dot,
  type FeedCustomer,
  type Movement,
} from "../src/matrix";
import { QUADRANTS } from "../src/overview";

function customer(id: string, respect: number, love: number, quadrant: FeedCustomer["quadrant"], churn = 0.05): FeedCustomer {
  return { customer_id: id, scores: { respect, love, fit: 0.5 }, churn_risk: churn, quadrant };
}

test("scores map onto the plot inside its margins", () => {
  expect(along(0)).toBe(MARGIN);
  expect(along(1)).toBe(SIZE - MARGIN);
  expect(along(LINE)).toBe(SIZE / 2);
});

test("respect runs across and love runs up", () => {
  expect(toDot(customer("c1", 1, 1, "lovemark"))).toMatchObject({ id: "c1", x: SIZE - MARGIN, y: MARGIN });
  const bottomLeft = toDot(customer("c2", 0, 0, "product"))!;
  expect([bottomLeft.x, bottomLeft.y]).toEqual([MARGIN, SIZE - MARGIN]);
});

test("a customer without scores, churn risk or a known quadrant has no dot", () => {
  expect(toDot({ customer_id: "c", scores: null, churn_risk: null, quadrant: null })).toBeNull();
  expect(toDot({ ...customer("c", 0.5, 0.5, "fad"), churn_risk: null })).toBeNull();
  expect(toDot(customer("c", 0.5, 0.5, "cult" as FeedCustomer["quadrant"]))).toBeNull();
  expect(toDot(customer("c", 0.5, 0.5, "toString" as FeedCustomer["quadrant"]))).toBeNull();
  expect(toDot(customer("c", Number.NaN, 0.5, "fad"))).toBeNull();
});

test("at risk is churn risk of 10% or more, as on the Overview", () => {
  expect(AT_RISK).toBe(0.1);
  expect(toDot(customer("c", 0.5, 0.5, "fad", 0.1))!.atRisk).toBe(true);
  expect(toDot(customer("c", 0.5, 0.5, "fad", 0.0999))!.atRisk).toBe(false);
});

test("counts list every quadrant, in the API's order", () => {
  const dots = [customer("a", 0.2, 0.8, "fad"), customer("b", 0.2, 0.9, "fad"), customer("c", 0.9, 0.9, "lovemark")]
    .map(toDot)
    .filter((d) => d !== null);
  expect(countByQuadrant(dots)).toEqual({ lovemark: 1, brand: 0, fad: 2, product: 0 });
  expect(countByQuadrant([])).toEqual({ lovemark: 0, brand: 0, fad: 0, product: 0 });
  expect(QUADRANTS).toEqual(["lovemark", "brand", "fad", "product"]);
});

test("a group is a quadrant, a least churn risk, or both", () => {
  const dot = toDot(customer("c", 0.2, 0.8, "fad", 0.12))!;
  expect(matches(dot, { quadrant: null, minChurn: 0 })).toBe(true);
  expect(matches(dot, { quadrant: "fad", minChurn: AT_RISK })).toBe(true);
  expect(matches(dot, { quadrant: "brand", minChurn: 0 })).toBe(false);
  expect(matches(dot, { quadrant: "fad", minChurn: 0.2 })).toBe(false);
});

test("a group can be the customers on one spot, whatever else is picked", () => {
  const dot = toDot(customer("c", 0.2, 0.8, "fad", 0.12))!;
  expect(matches(dot, { quadrant: null, minChurn: 0, ids: ["c", "d"] })).toBe(true);
  expect(matches(dot, { quadrant: "fad", minChurn: 0, ids: ["d"] })).toBe(false);
  expect(matches(dot, { quadrant: "fad", minChurn: 0, ids: null })).toBe(true);
});

// --- Spots, for the keyboard (T409) ---

function dotAt(id: string, respect: number, love: number): Dot {
  return toDot(customer(id, respect, love, "lovemark"))!;
}

//   love
//   0.9   a           b
//   0.5   c   d+e     f        (d and e share a spot)
//   0.1   g
//         0.1 0.5     0.9  respect
const DOTS = [
  dotAt("f", 0.9, 0.5),
  dotAt("e", 0.5, 0.5),
  dotAt("a", 0.1, 0.9),
  dotAt("g", 0.1, 0.1),
  dotAt("d", 0.5, 0.5),
  dotAt("b", 0.9, 0.9),
  dotAt("c", 0.1, 0.5),
];

const names = (spot: { dots: Dot[] } | null) => spot?.dots.map((d) => d.id).join("+") ?? null;

test("spots are the places dots are drawn, in reading order; a shared one holds its customers by id", () => {
  expect(spotsOf(DOTS).map(names)).toEqual(["a", "b", "c", "d+e", "f", "g"]);
  expect(spotsOf([])).toEqual([]);
});

test("an arrow moves to the nearest spot within 45 degrees of its direction", () => {
  const spots = spotsOf(DOTS);
  const at = (id: string) => spots.find((s) => s.dots.some((d) => d.id === id))!;
  expect(names(nextSpot(spots, at("c"), "ArrowRight"))).toBe("d+e");
  expect(names(nextSpot(spots, at("d"), "ArrowRight"))).toBe("f");
  expect(names(nextSpot(spots, at("d"), "ArrowLeft"))).toBe("c");
  expect(names(nextSpot(spots, at("c"), "ArrowUp"))).toBe("a");
  expect(names(nextSpot(spots, at("c"), "ArrowDown"))).toBe("g");
  // From a, b is straight across and d+e at 45 degrees: the straight one is nearer.
  expect(names(nextSpot(spots, at("a"), "ArrowRight"))).toBe("b");
  // d+e is up and to the left of f, at 45 degrees exactly: still in reach.
  expect(names(nextSpot(spots, at("f"), "ArrowLeft"))).toBe("d+e");
});

test("at an edge an arrow finds nothing", () => {
  const spots = spotsOf(DOTS);
  const a = spots[0]!;
  expect(nextSpot(spots, a, "ArrowUp")).toBeNull();
  expect(nextSpot(spots, a, "ArrowLeft")).toBeNull();
  expect(nextSpot(spots.slice(0, 1), a, "ArrowDown")).toBeNull();
});

test("every spot can be reached with the arrows from any other", () => {
  const spots = spotsOf(DOTS);
  for (const start of spots) {
    const seen = new Set([start.key]);
    const queue = [start];
    while (queue.length > 0) {
      const spot = queue.shift()!;
      for (const arrow of ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"] as const) {
        const next = nextSpot(spots, spot, arrow);
        if (next !== null && !seen.has(next.key)) {
          seen.add(next.key);
          queue.push(next);
        }
      }
    }
    expect(seen.size).toBe(spots.length);
  }
});

test("when a spot's customers move away, the nearest spot takes over", () => {
  const spots = spotsOf(DOTS);
  expect(names(nearestSpot(spots, along(0.52), SIZE - along(0.5)))).toBe("d+e");
  expect(nearestSpot([], 0, 0)).toBeNull();
});

test("a place on the plot reads back as its scores", () => {
  const { respect, love } = scoresAt(along(0.25), SIZE - along(0.75));
  expect(respect).toBeCloseTo(0.25, 10);
  expect(love).toBeCloseTo(0.75, 10);
});

test("movement totals: stayed, moved to another quadrant, scored for the first time", () => {
  const row = (lovemark = 0, brand = 0, fad = 0, product = 0) => ({ lovemark, brand, fad, product });
  const movement: Movement = {
    from: "2026-11-17",
    to: "2026-11-18",
    moves: {
      lovemark: row(140, 1),
      brand: row(1, 4),
      fad: row(0, 0, 11),
      product: row(0, 0, 0, 10),
      unscored: row(3, 0, 1),
    },
    customers: 171,
  };
  expect(moveTotals(movement)).toEqual({ stayed: 165, moved: 2, scored: 4 });
});

test("days are Bangkok's, whatever the computer's time zone", () => {
  // 18:30 UTC on the 18th is already 01:30 on the 19th in Bangkok.
  expect(bangkokDay(new Date("2026-11-18T18:30:00Z"))).toBe("2026-11-19");
  expect(bangkokDay(new Date("2026-11-18T18:30:00Z"), 1)).toBe("2026-11-18");
  expect(bangkokDay(new Date("2026-03-01T03:00:00Z"), 1)).toBe("2026-02-28");
});

test("Love as expected: exactly the live model's, never a near miss (T410)", () => {
  const live = 0.8291911026056588;
  const customers = [
    customer("on", 0.9, live, "lovemark"),
    customer("near", 0.9, 0.829, "lovemark"),
    customer("far", 0.2, 0.3, "product"),
    { customer_id: "unscored", scores: null, churn_risk: null, quadrant: null },
  ];
  expect([...lovesAsExpected(customers, live)]).toEqual(["on"]);
  expect(lovesAsExpected(customers, null).size).toBe(0);
});
