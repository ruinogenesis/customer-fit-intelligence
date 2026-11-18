// T402: the line chart's scale, and its crosshair on the keyboard.
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, test } from "vitest";

import { domain, LineChart, niceTicks } from "../src/LineChart";

afterEach(cleanup);

test("ticks fall on round steps inside the range", () => {
  expect(niceTicks(0.6, 0.85)).toEqual([0.6, 0.7, 0.8]);
  expect(niceTicks(0.031, 0.052)).toEqual([0.04, 0.05]);
  expect(niceTicks(0.038, 0.062)).toEqual([0.04, 0.05, 0.06]);
  expect(niceTicks(0, 100)).toEqual([0, 25, 50, 75, 100]);
});

test("the domain leaves a tenth of the range above and below", () => {
  expect(domain([1, 2, 3])).toEqual([0.8, 3.2]);
  expect(domain([0.5, 0.5])).toEqual([0.45, 0.55]); // a flat line still has room
});

const POINTS = [
  { label: "2026-11-15", value: 0.71 },
  { label: "2026-11-16", value: 0.74 },
  { label: "2026-11-17", value: 0.73 },
];

function chart() {
  render(<LineChart title="Average fit" points={POINTS} format={(v) => v.toFixed(3)} />);
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
  const { container } = render(<LineChart title="Empty" points={[]} format={String} />);
  expect(container.innerHTML).toBe("");
});
