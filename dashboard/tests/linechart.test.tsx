// T402: the line chart's scale and axis, and its crosshair on the keyboard.
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, test } from "vitest";

import { digitsFor, domain, LineChart, niceTicks, tickStep } from "../src/LineChart";

afterEach(cleanup);

test("ticks fall on round steps inside the range", () => {
  expect(niceTicks(0.6, 0.85)).toEqual([0.6, 0.7, 0.8]);
  expect(niceTicks(0.031, 0.052)).toEqual([0.04, 0.05]);
  expect(niceTicks(0.038, 0.062)).toEqual([0.04, 0.05, 0.06]);
  expect(niceTicks(0, 100)).toEqual([0, 25, 50, 75, 100]);
});

test("ticks are labelled with as many decimals as their step needs", () => {
  expect(tickStep(0.4754, 0.4826)).toBe(0.002);
  expect([0.002, 0.0025, 0.02, 0.5, 2.5, 5, 25].map(digitsFor)).toEqual([3, 4, 2, 1, 1, 0, 0]);
});

test("the domain leaves a tenth of the range above and below", () => {
  expect(domain([1, 2, 3], 0)).toEqual([0.8, 3.2]);
  expect(domain([0.5, 0.5], 0)).toEqual([0.45, 0.55]); // a flat line still has room
});

test("the domain is at least the minimum span wide, around the values", () => {
  const [lo, hi] = domain([0.48, 0.48], 0.05);
  expect(lo).toBeCloseTo(0.45, 12);
  expect(hi).toBeCloseTo(0.51, 12);
  const [wideLo, wideHi] = domain([0.4, 0.6], 0.05); // wider than the minimum: as before
  expect(wideLo).toBeCloseTo(0.38, 12);
  expect(wideHi).toBeCloseTo(0.62, 12);
});

// Dev's first 28 days (T402): one average every day, apart from the
// floating-point noise of BigQuery summing in a different order each night.
// Scaled to its own range, the noise filled the chart and the ticks fell off it.
const NOISE = 2 ** -54; // one step between doubles near 0.48
const FLAT = [0, 2, 3, 2, 0, 1, 5].map((k, i) => ({
  label: `2026-11-1${i}`,
  value: 0.48213457 + k * NOISE,
}));

function ticksOf(svg: Element): { label: string | null; y: number }[] {
  return [...svg.querySelectorAll(".tick text")].map((t) => ({
    label: t.textContent,
    y: Number(t.getAttribute("y")),
  }));
}

test("a flat line with floating-point noise draws flat, its ticks inside the plot", () => {
  render(
    <LineChart
      title="Average fit"
      points={FLAT}
      minSpan={0.05}
      format={(v) => v.toFixed(3)}
      axis={(v, step) => v.toFixed(digitsFor(step))}
    />,
  );
  const svg = screen.getByRole("img");
  const heights = svg.querySelector(".series")!.getAttribute("d")!.match(/,[-\d.e]+/g)!;
  expect(new Set(heights.map((h) => Number(h.slice(1)).toFixed(6))).size).toBe(1);
  const ticks = ticksOf(svg);
  expect(ticks.map((t) => t.label)).toEqual(["0.46", "0.48", "0.50"]);
  for (const { y } of ticks) {
    expect(y).toBeGreaterThanOrEqual(12); // the plot runs from y = 12 to 136
    expect(y).toBeLessThanOrEqual(136);
  }
});

test("on a narrow range the axis labels stay distinct", () => {
  const points = [0.476, 0.478, 0.482, 0.479].map((value, i) => ({ label: `d${i}`, value }));
  render(
    <LineChart
      title="Average fit"
      points={points}
      minSpan={0}
      format={(v) => v.toFixed(3)}
      axis={(v, step) => v.toFixed(digitsFor(step))}
    />,
  );
  expect(ticksOf(screen.getByRole("img")).map((t) => t.label)).toEqual([
    "0.476",
    "0.478",
    "0.480",
    "0.482",
  ]);
});

const POINTS = [
  { label: "2026-11-15", value: 0.71 },
  { label: "2026-11-16", value: 0.74 },
  { label: "2026-11-17", value: 0.73 },
];

function chart() {
  render(<LineChart title="Average fit" points={POINTS} minSpan={0} format={(v) => v.toFixed(3)} />);
  return screen.getByRole("img");
}

test("the chart names its span and latest value, which is labelled at the end", () => {
  const svg = chart();
  expect(svg.getAttribute("aria-label")).toBe(
    "Average fit, 2026-11-15 to 2026-11-17: latest 0.730. Arrow keys step through the days.",
  );
  expect(svg.querySelector("text.end")?.textContent).toBe("0.730");
  expect(screen.queryByRole("status")).toBeNull();
});

test("on focus the crosshair shows the latest day; arrows and Home step through", () => {
  const svg = chart();
  fireEvent.focus(svg);
  expect(screen.getByRole("status").textContent).toBe("0.730 2026-11-17");
  fireEvent.keyDown(svg, { key: "ArrowLeft" });
  expect(screen.getByRole("status").textContent).toBe("0.740 2026-11-16");
  fireEvent.keyDown(svg, { key: "Home" });
  fireEvent.keyDown(svg, { key: "ArrowLeft" }); // stays on the first
  expect(screen.getByRole("status").textContent).toBe("0.710 2026-11-15");
  fireEvent.blur(svg);
  expect(screen.queryByRole("status")).toBeNull();
});

test("no points, no chart", () => {
  const { container } = render(<LineChart title="Empty" points={[]} minSpan={0} format={String} />);
  expect(container.innerHTML).toBe("");
});
