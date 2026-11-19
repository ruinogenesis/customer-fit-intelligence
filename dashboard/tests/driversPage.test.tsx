// T405: the Drivers screen: each input's share of fit over every order, then
// restaurants or couriers worst first, a row opening all eight inputs; the
// grouping, span and minimum kept in the URL.
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { createMemoryRouter } from "react-router";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { ApiError } from "../src/api";
import { App } from "../src/App";
import { INPUTS, type Input } from "../src/customers";
import type { DriverGroup, Drivers } from "../src/drivers";
import { violations } from "./axe";
import { FakeApi, FakeAuth } from "./fakes";

// 10:00 on 2026-11-19 in Bangkok: the last 28 days start on 2026-10-23.
const NOW = new Date("2026-11-19T03:00:00Z");

const inputs = (values: Partial<Record<Input, number>>) =>
  Object.fromEntries(INPUTS.map((input) => [input, values[input] ?? 0])) as Record<Input, number>;

const OVERALL = inputs({ x1: -0.09, x2: -0.008, x3: -0.01, x5: -0.008, x6: -0.026, x8: -0.006 });

/** A group as the API answers it: drag from the excess, drivers worst first. */
function group(id: string, orders: number, fit: number, excess: Partial<Record<Input, number>>): DriverGroup {
  const all = inputs(excess);
  const drag = inputs(Object.fromEntries(INPUTS.map((input) => [input, OVERALL[input] + all[input]])));
  const drivers = INPUTS.filter((input) => all[input] < 0)
    .sort((a, b) => all[a] - all[b])
    .slice(0, 3);
  return { id, orders, fit, drag, excess: all, drivers };
}

function answer(
  by: Drivers["by"],
  from: string,
  to: string,
  minOrders: number,
  groups: DriverGroup[],
  orders = 120,
): Drivers {
  return {
    by,
    from,
    to,
    min_orders: minOrders,
    overall: { orders, fit: orders === 0 ? null : 0.595, drag: orders === 0 ? inputs({}) : OVERALL },
    groups,
  };
}

const RESTAURANTS = [
  group("rest-02", 45, 0.453, { x1: 0.026, x2: 0.012, x3: -0.0043, x5: 0.003, x6: -0.178, x8: -0.0002 }),
  group("rest-01", 42, 0.513, { x1: -0.081, x2: 0.007, x3: -0.027, x5: 0.007, x6: 0.019, x8: -0.0067 }),
  group("rest-05", 7, 0.7, { x1: 0.05, x3: 0.01, x6: 0.02, x8: 0.001 }),
];
const COURIERS = [group("courier-02", 30, 0.52, { x5: -0.06, x1: -0.01 })];

const path = (by: string, from: string, to: string, min = 5) =>
  `/drivers?by=${by}&from=${from}&to=${to}&min_orders=${min}&limit=200`;
const LAST_28 = path("restaurant", "2026-10-23", "2026-11-19");

function fakeApi(): FakeApi {
  const api = new FakeApi();
  api.answers = {
    [LAST_28]: answer("restaurant", "2026-10-23", "2026-11-19", 5, RESTAURANTS),
    [path("courier", "2026-10-23", "2026-11-19")]: answer("courier", "2026-10-23", "2026-11-19", 5, COURIERS),
    [path("courier", "2026-11-13", "2026-11-19")]: answer("courier", "2026-11-13", "2026-11-19", 5, COURIERS, 40),
    [path("courier", "2026-11-13", "2026-11-19", 10)]: answer("courier", "2026-11-13", "2026-11-19", 10, COURIERS, 40),
    [path("courier", "2026-11-01", "2026-11-19", 10)]: answer("courier", "2026-11-01", "2026-11-19", 10, COURIERS, 80),
    [path("courier", "2026-01-01", "2026-04-01", 1)]: answer("courier", "2026-01-01", "2026-04-01", 1, COURIERS, 900),
  };
  return api;
}

beforeEach(() => {
  vi.useFakeTimers({ now: NOW });
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

async function settle() {
  for (let i = 0; i < 4; i++) {
    await act(async () => {
      await vi.dynamicImportSettled();
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
  }
}

async function open(at = "/drivers", api = fakeApi()) {
  const auth = new FakeAuth();
  let router!: ReturnType<typeof createMemoryRouter>;
  render(
    <App
      auth={auth}
      api={api}
      createRouter={(routes) => (router = createMemoryRouter(routes, { initialEntries: [at] }))}
    />,
  );
  auth.answer({ email: "viewer@example.com", role: "viewer" });
  await settle();
  return { api, search: () => router.state.location.search };
}

function pressed(name: string): string[] {
  return within(screen.getByRole("group", { name }))
    .getAllByRole("button")
    .map((b) => `${b.textContent}${b.getAttribute("aria-pressed") === "true" ? " (picked)" : ""}`);
}

function spans(): string[] {
  return [...document.querySelectorAll(".days button")].map(
    (b) => `${b.textContent}${b.getAttribute("aria-pressed") === "true" ? " (picked)" : ""}`,
  );
}

function ranked(): string[] {
  return [...document.querySelectorAll("table.ranking > tbody > tr:not(.detail)")].map((r) => r.textContent ?? "");
}

function bars(label: string): { amount: string; bar: string | null; width: string | null }[] {
  return [...screen.getByRole("table", { name: label }).querySelectorAll("tr")].map((row) => {
    const bar = row.querySelector<HTMLElement>(".bar");
    return {
      amount: row.querySelector(".amount")?.textContent ?? "",
      bar: bar?.className ?? null,
      width: bar?.style.width ?? null,
    };
  });
}

const amounts = (label: string) => bars(label).map((b) => b.amount);
const EVERY_ORDER = "Fit over every order: each input's average share";

// --- What it shows ---

test("it opens on the last 28 days by restaurant: every order's split, then the worst first", async () => {
  const { api, search } = await open();
  expect(api.asked(LAST_28)).toBe(1);
  expect(search()).toBe("");
  expect(pressed("Group by")).toEqual(["Restaurants (picked)", "Couriers"]);
  expect(spans()).toEqual(["7 days", "28 days (picked)", "90 days"]);
  expect((screen.getByLabelText(/Orders placed from/) as HTMLInputElement).value).toBe("2026-10-23");
  expect((screen.getByLabelText(/^to/) as HTMLInputElement).value).toBe("2026-11-19");
  expect((screen.getByLabelText(/At least/) as HTMLSelectElement).value).toBe("5");

  expect(screen.getByText("120 orders placed from 2026-10-23 to 2026-11-19, average fit 0.595.")).toBeTruthy();
  expect(amounts(EVERY_ORDER)).toEqual(["−0.090", "−0.008", "−0.010", "0.000", "−0.008", "−0.026", "0.000", "−0.006"]);
  expect(bars(EVERY_ORDER)[0]).toEqual({ amount: "−0.090", bar: "bar left worse", width: "50%" });
  expect(bars(EVERY_ORDER)[3]).toEqual({ amount: "0.000", bar: null, width: null });

  expect(
    screen.getByText(
      "3 restaurants have at least 5 orders, worst first: by the input that pulls fit down most, " +
        "compared with every order. Click one to see all eight inputs.",
    ),
  ).toBeTruthy();
  expect(ranked()).toEqual([
    "rest-02450.453Food & selection −0.178, Fee load −0.004",
    "rest-01420.513Speed vs promise −0.081, Fee load −0.027, Recovery handling −0.007",
    "rest-0570.700Nothing",
  ]);
  expect(screen.getByText(/^Handing a fix to its owner isn't built yet\./)).toBeTruthy();
});

test("a row opens all eight inputs against every order; one row at a time", async () => {
  await open();
  const label = "rest-02: each input's share of fit, minus its share over every order";
  fireEvent.click(screen.getByRole("button", { name: "rest-02" }));
  expect(screen.getByRole("button", { name: "rest-02" }).getAttribute("aria-expanded")).toBe("true");
  expect(screen.getByText("45 orders, fit 0.453; over every order, 0.595.")).toBeTruthy();
  expect(amounts(label)).toEqual(["+0.026", "+0.012", "−0.004", "0.000", "+0.003", "−0.178", "0.000", "0.000"]);
  expect(bars(label)[5]).toEqual({ amount: "−0.178", bar: "bar left worse", width: "50%" });
  // 0.026 of 0.178, half the width: about 7.3%.
  expect(bars(label)[0]).toEqual({
    amount: "+0.026",
    bar: "bar right better",
    width: expect.stringMatching(/^7\.30\d*%$/),
  });

  fireEvent.click(screen.getByRole("button", { name: "rest-01" }));
  expect(screen.queryByRole("table", { name: label })).toBeNull();
  expect(screen.getByRole("button", { name: "rest-02" }).getAttribute("aria-expanded")).toBe("false");
  expect(screen.getByText("42 orders, fit 0.513; over every order, 0.595.")).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "rest-01" }));
  expect(document.querySelector("tr.detail")).toBeNull();
});

test("axe finds nothing on the drivers screen, with a row open (T409)", async () => {
  await open();
  fireEvent.click(screen.getByRole("button", { name: "rest-02" }));
  expect(await violations()).toEqual([]);
});

// --- The URL ---

test("couriers, a span and a minimum go into the URL and the question", async () => {
  const { api, search } = await open();
  fireEvent.click(screen.getByRole("button", { name: "Couriers" }));
  await settle();
  expect(search()).toBe("?by=courier");
  expect(pressed("Group by")).toEqual(["Restaurants", "Couriers (picked)"]);
  expect(screen.getByRole("columnheader", { name: "Courier" })).toBeTruthy();
  expect(screen.getByText(/^1 courier has at least 5 orders, worst first/)).toBeTruthy();
  expect(ranked()).toEqual(["courier-02300.520Courier conduct −0.060, Speed vs promise −0.010"]);

  fireEvent.click(screen.getByRole("button", { name: "7 days" }));
  await settle();
  expect(search()).toBe("?by=courier&from=2026-11-13&to=2026-11-19");
  expect(spans()).toEqual(["7 days (picked)", "28 days", "90 days"]);
  expect(screen.getByText("40 orders placed from 2026-11-13 to 2026-11-19, average fit 0.595.")).toBeTruthy();

  fireEvent.change(screen.getByLabelText(/At least/), { target: { value: "10" } });
  await settle();
  expect(search()).toBe("?by=courier&from=2026-11-13&to=2026-11-19&min_orders=10");
  expect(screen.getByText(/^1 courier has at least 10 orders/)).toBeTruthy();

  fireEvent.change(screen.getByLabelText(/Orders placed from/), { target: { value: "2026-11-01" } });
  await settle();
  expect(search()).toBe("?by=courier&from=2026-11-01&to=2026-11-19&min_orders=10");
  expect(spans()).toEqual(["7 days", "28 days", "90 days"]);
  expect(screen.getByText(/^80 orders placed from 2026-11-01/)).toBeTruthy();
  expect(api.calls.filter((call) => call.startsWith("GET /drivers"))).toHaveLength(5);
});

test("a link with a span and a minimum opens on them", async () => {
  const { api } = await open("/drivers?by=courier&from=2026-01-01&to=2026-04-01&min_orders=1");
  expect(api.asked(path("courier", "2026-01-01", "2026-04-01", 1))).toBe(1);
  expect((screen.getByLabelText(/Orders placed from/) as HTMLInputElement).value).toBe("2026-01-01");
  expect((screen.getByLabelText(/^to/) as HTMLInputElement).value).toBe("2026-04-01");
  expect(spans()).toEqual(["7 days", "28 days", "90 days"]);
  expect(screen.getByText(/^900 orders placed from 2026-01-01 to 2026-04-01/)).toBeTruthy();
  const minimum = screen.getByLabelText(/At least/) as HTMLSelectElement;
  expect(minimum.value).toBe("1");
  expect([...minimum.options].map((o) => o.textContent)).toEqual(["1 order", "5 orders", "10 orders", "20 orders"]);
});

test("a minimum not in the list is added to it; parts it cannot read are left at their defaults", async () => {
  const api = fakeApi();
  api.answers[path("restaurant", "2026-10-23", "2026-11-19", 3)] = answer(
    "restaurant",
    "2026-10-23",
    "2026-11-19",
    3,
    RESTAURANTS,
  );
  await open("/drivers?min_orders=3", api);
  const minimum = screen.getByLabelText(/At least/) as HTMLSelectElement;
  expect(minimum.value).toBe("3");
  expect([...minimum.options].map((o) => o.value)).toEqual(["1", "3", "5", "10", "20"]);
  cleanup();
  const again = await open("/drivers?by=bus&from=yesterday&min_orders=0");
  expect(again.api.asked(LAST_28)).toBe(1);
});

// --- Nothing to rank, and refusals ---

test("a span with no orders, and a minimum no group reaches, each say so", async () => {
  const api = fakeApi();
  api.answers[LAST_28] = answer("restaurant", "2026-10-23", "2026-11-19", 5, [], 0);
  api.answers[path("restaurant", "2026-10-23", "2026-11-19", 20)] = answer(
    "restaurant",
    "2026-10-23",
    "2026-11-19",
    20,
    [],
  );
  await open("/drivers", api);
  expect(screen.getByText("No orders were placed from 2026-10-23 to 2026-11-19.")).toBeTruthy();
  expect(document.querySelector("table")).toBeNull();
  fireEvent.change(screen.getByLabelText(/At least/), { target: { value: "20" } });
  await settle();
  expect(screen.getByText("No restaurant has at least 20 orders in this span. Lower the minimum to see more.")).toBeTruthy();
  expect(screen.getByRole("table", { name: EVERY_ORDER })).toBeTruthy();
  expect(document.querySelector("table.ranking")).toBeNull();
});

test("when the API's most groups come back, it says the list stops there", async () => {
  const api = fakeApi();
  const many = Array.from({ length: 200 }, (_, i) => group(`rest-${String(i).padStart(3, "0")}`, 5, 0.6, { x1: -0.001 }));
  api.answers[LAST_28] = answer("restaurant", "2026-10-23", "2026-11-19", 5, many, 1000);
  await open("/drivers", api);
  expect(ranked()).toHaveLength(200);
  expect(screen.getByText("Showing the worst 200; raise the minimum to narrow it.")).toBeTruthy();
});

test("a first day after the last is not asked for; the API's refusals are shown", async () => {
  const { api } = await open("/drivers?from=2026-11-20&to=2026-11-19");
  expect(screen.getByRole("alert").textContent).toBe("Pick a first day on or before the last.");
  expect(api.calls.some((call) => call.startsWith("GET /drivers"))).toBe(false);
  cleanup();

  const refusing = fakeApi();
  refusing.answers[LAST_28] = new ApiError(
    503,
    "This API reads no BigQuery history (HISTORY is unset), so it cannot rank drivers.",
  );
  await open("/drivers", refusing);
  expect(screen.getByRole("alert").textContent).toBe(
    "This API reads no BigQuery history (HISTORY is unset), so it cannot rank drivers.",
  );
});
