// T405: the Drivers screen's view in the URL, the question it asks the API,
// and which drivers are worth naming.
import { expect, test } from "vitest";

import { INPUTS, type Input } from "../src/customers";
import { driversPath, namedDrivers, readView, spanDays, type DriverGroup } from "../src/drivers";
import { addDays } from "../src/matrix";

const TODAY = "2026-11-19";

test("with nothing in the URL, the view is the last 28 days by restaurant, 5 orders or more", () => {
  expect(readView(new URLSearchParams(), TODAY)).toEqual({
    by: "restaurant",
    from: "2026-10-23",
    to: "2026-11-19",
    minOrders: 5,
  });
});

test("the URL's parts are kept; a last day alone gets the 28 days before it", () => {
  expect(readView(new URLSearchParams("by=courier&to=2026-04-01&min_orders=10"), TODAY)).toEqual({
    by: "courier",
    from: "2026-03-05",
    to: "2026-04-01",
    minOrders: 10,
  });
  expect(readView(new URLSearchParams("from=2026-01-01&to=2026-04-01&min_orders=1"), TODAY)).toEqual({
    by: "restaurant",
    from: "2026-01-01",
    to: "2026-04-01",
    minOrders: 1,
  });
});

test("parts it cannot read fall back to their defaults", () => {
  const unreadable = "by=bus&from=1/1/2026&to=soon&min_orders=2.5";
  expect(readView(new URLSearchParams(unreadable), TODAY)).toEqual(readView(new URLSearchParams(), TODAY));
  for (const min of ["0", "-3", "many", ""]) {
    expect(readView(new URLSearchParams({ min_orders: min }), TODAY).minOrders).toBe(5);
  }
});

test("the question names every part, and asks for the API's most groups", () => {
  expect(driversPath({ by: "courier", from: "2026-01-01", to: "2026-04-01", minOrders: 1 })).toBe(
    "/drivers?by=courier&from=2026-01-01&to=2026-04-01&min_orders=1&limit=200",
  );
});

test("spans count both days; days shift across months and years", () => {
  expect(spanDays("2026-01-01", "2026-04-01")).toBe(91);
  expect(spanDays("2026-10-23", "2026-11-19")).toBe(28);
  expect(spanDays("2026-11-19", "2026-11-19")).toBe(1);
  expect(addDays("2026-03-01", -1)).toBe("2026-02-28");
  expect(addDays("2026-12-31", 1)).toBe("2027-01-01");
});

function group(excess: Partial<Record<Input, number>>, drivers: Input[]): DriverGroup {
  const all = Object.fromEntries(INPUTS.map((input) => [input, excess[input] ?? 0])) as Record<Input, number>;
  return { id: "rest-02", orders: 45, fit: 0.453, drag: all, excess: all, drivers };
}

test("a driver is named only when its excess shows at three decimals", () => {
  expect(namedDrivers(group({ x6: -0.178, x3: -0.0043, x8: -0.0002 }, ["x6", "x3", "x8"]))).toEqual([
    "x6",
    "x3",
  ]);
  expect(namedDrivers(group({ x1: -0.0005 }, ["x1"]))).toEqual(["x1"]);
  expect(namedDrivers(group({ x1: 0.05 }, []))).toEqual([]);
});
