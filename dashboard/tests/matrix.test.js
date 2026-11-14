// T206: the love matrix's geometry and counts (run: node --test).

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  LINE,
  MARGIN,
  QUADRANTS,
  SIZE,
  along,
  countByQuadrant,
  toDot,
} from "../public/matrix.js";

test("scores map onto the plot inside its margins", () => {
  assert.equal(along(0), MARGIN);
  assert.equal(along(1), SIZE - MARGIN);
  assert.equal(along(LINE), SIZE / 2);
});

test("respect runs across and love runs up", () => {
  const topRight = toDot("c1", {
    scores: { respect: 1, love: 1 },
    quadrant: "lovemark",
  });
  assert.deepEqual(topRight, {
    id: "c1",
    x: SIZE - MARGIN,
    y: MARGIN,
    quadrant: "lovemark",
  });
  const bottomLeft = toDot("c2", {
    scores: { respect: 0, love: 0 },
    quadrant: "product",
  });
  assert.deepEqual([bottomLeft.x, bottomLeft.y], [MARGIN, SIZE - MARGIN]);
});

test("a customer without scores or a known quadrant has no dot", () => {
  const half = { respect: 0.5, love: 0.5 };
  assert.equal(toDot("c", { scores: null, quadrant: null }), null);
  assert.equal(toDot("c", { quadrant: "fad" }), null);
  assert.equal(toDot("c", { scores: half, quadrant: "cult" }), null);
  assert.equal(toDot("c", { scores: half, quadrant: "toString" }), null);
  const text = { respect: "0.5", love: 0.5 };
  assert.equal(toDot("c", { scores: text, quadrant: "fad" }), null);
});

test("counts list every quadrant", () => {
  const dots = [{ quadrant: "fad" }, { quadrant: "fad" }, { quadrant: "lovemark" }];
  assert.deepEqual(countByQuadrant(dots), {
    lovemark: 1,
    brand: 0,
    fad: 2,
    product: 0,
  });
  assert.deepEqual(countByQuadrant([]), { lovemark: 0, brand: 0, fad: 0, product: 0 });
});

test("the quadrants are the API's, in its order", () => {
  assert.deepEqual(Object.keys(QUADRANTS), ["lovemark", "brand", "fad", "product"]);
});
