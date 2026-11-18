// T206, in the app since T403: the love matrix's geometry, dots and counts.
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
  SIZE,
  toDot,
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
