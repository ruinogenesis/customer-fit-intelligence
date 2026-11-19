// T404: the Customers list and one customer's profile: their scores now and
// over time, the exact split of what moved a score and what changed since the
// scoring before, their orders, and checking again while visible.
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { createMemoryRouter } from "react-router";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { ApiError } from "../src/api";
import { App } from "../src/App";
import { HISTORY_PAGE, NEW_PAGE, ORDERS, REFRESH_MS } from "../src/CustomerPage";
import {
  baht,
  changeOf,
  delivery,
  INPUTS,
  mergeScorings,
  signed,
  splitOf,
  type CustomerState,
  type Input,
  type Order,
  type Score,
  type Scoring,
} from "../src/customers";
import type { Role } from "../src/roles";
import { violations } from "./axe";
import { FakeApi, FakeAuth } from "./fakes";

const ID = "sim404-000001";
const PROFILE = `/customers/${ID}`;
const ALL = `${PROFILE}/scores?limit=${HISTORY_PAGE}`;
const NEW = `${PROFILE}/scores?limit=${NEW_PAGE}`;
const ORDER_PAGE = `${PROFILE}/orders?limit=${ORDERS}`;

// "As expected" (every gap 0) under the model: the same for every scoring
// it makes. A scoring's value is that plus its parts.
const EXPECTED: Record<Score, number> = {
  fit: 0.743,
  love: 0.8,
  respect: 0.78,
  trust: 0.76,
  convenience: 0.7,
  churn_risk: 0.012,
};

function parts(given: Partial<Record<Input, number>>, scale = 1): Record<Input, number> {
  return Object.fromEntries(INPUTS.map((x) => [x, (given[x] ?? 0) * scale])) as Record<Input, number>;
}

/** A scoring whose fit moved by fitParts; the other scores by a share of them. */
function scoring(event: string, order: string, at: string, fitParts: Partial<Record<Input, number>>): Scoring {
  const scales: Record<Score, number> = {
    fit: 1,
    love: 0.5,
    respect: 0.8,
    trust: 0.6,
    convenience: 1.2,
    churn_risk: -0.05, // a worse gap raises churn risk
  };
  const contributions = Object.fromEntries(
    Object.entries(scales).map(([score, scale]) => [score, parts(fitParts, scale)]),
  ) as Record<Score, Record<Input, number>>;
  const value = (score: Score) =>
    EXPECTED[score] + INPUTS.reduce((sum, x) => sum + contributions[score][x], 0);
  return {
    event_id: event,
    order_id: order,
    at,
    scores: {
      fit: value("fit"),
      love: value("love"),
      respect: value("respect"),
      trust: value("trust"),
      convenience: value("convenience"),
    },
    churn_risk: value("churn_risk"),
    quadrant: "lovemark",
    contributions,
    drivers: ["x1", "x6", "x3"],
    model_version: "t112-sim",
  };
}

const ORDER_1 = `${ID}-001`;
const ORDER_2 = `${ID}-002`;

// Newest first, as the API lists them. The oldest was scored before T316 by
// an older model, so it carries no split.
const LATEST = scoring("e3", ORDER_2, "2026-03-01T12:41:48Z", {
  x1: -0.114,
  x2: 0.012,
  x3: 0.042,
  x5: -0.019,
  x6: -0.059,
});
const BEFORE = scoring("e2", ORDER_2, "2026-03-01T12:02:00Z", {
  x1: -0.07,
  x2: 0.012,
  x3: 0.042,
  x5: -0.019,
  x6: -0.059,
});
const OLDEST: Scoring = (() => {
  const old = scoring("e1", ORDER_1, "2026-02-20T05:00:00Z", { x3: 0.02 });
  delete old.contributions;
  delete old.drivers;
  return { ...old, model_version: "t100-old" };
})();

function state(latest: Scoring): CustomerState {
  return {
    customer_id: ID,
    latest_order_id: latest.order_id,
    latest_placed_at: "2026-03-01T12:01:33Z",
    scores: latest.scores,
    churn_risk: latest.churn_risk,
    quadrant: latest.quadrant,
    model_version: latest.model_version,
  };
}

function order(id: string, placed: string, promised: string, delivered: string | null, more: Partial<Order> = {}): Order {
  return {
    order_id: id,
    customer_id: ID,
    restaurant_id: "rest-23",
    placed_at: placed,
    promised_eta: promised,
    basket_value: 19118,
    fees: 1044,
    promo_used: true,
    courier_id: delivered === null ? null : "courier-31",
    delivered_at: delivered,
    issues: [],
    ratings: {},
    gaps: {},
    ...more,
  };
}

const ORDERS_NOW = [
  order(ORDER_2, "2026-03-01T12:01:33Z", "2026-03-01T12:30:00Z", "2026-03-01T12:41:00Z", {
    ratings: { food: 4, courier: 4 },
  }),
  order(ORDER_1, "2026-02-20T04:30:00Z", "2026-02-20T05:00:00Z", "2026-02-20T04:56:00Z", {
    promo_used: false,
    issues: [{ issue_type: "cold", at: "2026-02-20T05:10:00Z" }],
  }),
];

function fakeApi(): FakeApi {
  const api = new FakeApi();
  api.answers = {
    "/customers?limit=50": {
      customers: [
        { customer_id: ID, quadrant: "lovemark", churn_risk: LATEST.churn_risk, scores: LATEST.scores },
        { customer_id: "sim404-000002", quadrant: null, churn_risk: null, scores: null },
      ],
      next_cursor: "page-2",
    },
    "/customers?limit=50&cursor=page-2": {
      customers: [{ customer_id: "sim404-000003", quadrant: "fad", churn_risk: 0.01, scores: { fit: 0.4 } }],
      next_cursor: null,
    },
    "/customers?limit=50&quadrant=fad": {
      customers: [{ customer_id: "sim404-000003", quadrant: "fad", churn_risk: 0.01, scores: { fit: 0.4 } }],
      next_cursor: null,
    },
    [PROFILE]: state(LATEST),
    [ALL]: { scores: [LATEST, BEFORE, OLDEST], next_cursor: null },
    [ORDER_PAGE]: { orders: ORDERS_NOW, next_cursor: null },
  };
  return api;
}

let visibility: DocumentVisibilityState = "visible";

beforeEach(() => {
  vi.useFakeTimers();
  visibility = "visible";
  vi.spyOn(document, "visibilityState", "get").mockImplementation(() => visibility);
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

async function wait(ms = 0) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

async function open(path: string, api = fakeApi(), role: Role = "viewer") {
  const auth = new FakeAuth();
  render(
    <App
      auth={auth}
      api={api}
      createRouter={(routes) => createMemoryRouter(routes, { initialEntries: [path] })}
    />,
  );
  auth.answer({ email: `${role}@example.com`, role });
  for (let i = 0; i < 4; i++) {
    await act(async () => {
      await vi.dynamicImportSettled();
    });
    await wait();
  }
  return api;
}

function rows(table: HTMLElement): string[] {
  return within(table).getAllByRole("row").map((r) => r.textContent ?? "");
}

function bars(): { input: string; amount: string; bar: string | null }[] {
  const table = screen.getByRole("table", { name: /each input's/ });
  return within(table)
    .getAllByRole("row")
    .map((r) => ({
      input: r.querySelector("th")?.textContent ?? "",
      amount: r.querySelector(".amount")?.textContent ?? "",
      bar: r.querySelector(".bar")?.className ?? null,
    }));
}

function sum(): string {
  return document.querySelector(".sum")?.textContent ?? "";
}

// --- The sums ---

test("a split adds up to its score exactly, from as expected", () => {
  const split = splitOf(LATEST, "fit")!;
  expect(split.expected).toBeCloseTo(0.743, 12);
  expect(split.value).toBe(LATEST.scores.fit);
  expect(INPUTS.reduce((s, x) => s + split.parts[x], split.expected)).toBeCloseTo(split.value, 12);
  expect(splitOf(OLDEST, "fit")).toBeNull(); // scored before T316
});

test("a change is the difference of two splits, and says when the model changed", () => {
  const change = changeOf(BEFORE, LATEST, "fit")!;
  expect(change.to - change.from).toBeCloseTo(-0.044, 12);
  expect(change.parts.x1).toBeCloseTo(-0.044, 12);
  expect(change.parts.x3).toBeCloseTo(0, 12);
  expect(change.sameModel).toBe(true);
  expect(changeOf(OLDEST, BEFORE, "fit")).toBeNull();
  const newer = { ...LATEST, model_version: "t500-new" };
  expect(changeOf(BEFORE, newer, "fit")!.sameModel).toBe(false);
});

test("new scorings merge in once each, newest first", () => {
  const LATER = scoring("e4", ORDER_2, "2026-03-02T09:00:00Z", { x1: 0.01 });
  expect(mergeScorings([LATEST, BEFORE], [LATER, LATEST]).map((s) => s.event_id)).toEqual(["e4", "e3", "e2"]);
});

test("money, delivery and signs read as people say them", () => {
  expect(baht(19118)).toBe("฿191.18");
  expect(delivery(ORDERS_NOW[0]!)).toBe("11 min late");
  expect(delivery(ORDERS_NOW[1]!)).toBe("4 min early");
  expect(delivery(order("o", "2026-03-01T12:00:00Z", "2026-03-01T12:30:00Z", null))).toBe("not delivered");
  expect([signed(0.042), signed(-0.114), signed(0.00004)]).toEqual(["+0.042", "−0.114", "0.000"]);
});

// --- The list ---

test("the list is riskiest first, a page at a time, each customer a link to their profile", async () => {
  await open("/customers");
  const table = screen.getByRole("table");
  expect(rows(table)).toEqual([
    "CustomerChurn riskFitQuadrant",
    `${ID}1.9%0.605Lovemark`,
    "sim404-000002——not scored",
  ]);
  expect(within(table).getByRole("link", { name: ID }).getAttribute("href")).toBe(PROFILE);
  fireEvent.click(screen.getByRole("button", { name: "Load more" }));
  await wait();
  expect(rows(screen.getByRole("table")).at(-1)).toBe("sim404-0000031.0%0.400Fad");
  expect(screen.queryByRole("button", { name: "Load more" })).toBeNull();
});

test("axe finds nothing on the customers list (T409)", async () => {
  await open("/customers");
  expect(await violations()).toEqual([]);
});

test("the quadrant filter asks the API for that quadrant only", async () => {
  const api = await open("/customers");
  fireEvent.change(screen.getByLabelText(/Quadrant/), { target: { value: "fad" } });
  await wait();
  expect(api.asked("/customers?limit=50&quadrant=fad")).toBe(1);
  expect(rows(screen.getByRole("table"))).toEqual(["CustomerChurn riskFitQuadrant", "sim404-0000031.0%0.400Fad"]);
});

test("a link opens the profile", async () => {
  await open("/customers");
  fireEvent.click(screen.getByRole("link", { name: ID }));
  for (let i = 0; i < 4; i++) {
    // The profile is its own chunk.
    await act(async () => {
      await vi.dynamicImportSettled();
    });
    await wait();
  }
  expect(screen.getByRole("heading", { level: 1 }).textContent).toBe(`Customer ${ID}`);
});

// --- The profile ---

test("the profile shows the customer's scores now", async () => {
  await open(PROFILE);
  const tiles = [...document.querySelectorAll(".tile")].map((t) =>
    [...t.children].map((c) => c.textContent).join(" | "),
  );
  expect(tiles).toEqual([
    "Fit | 0.605",
    "Love | 0.731",
    "Respect | 0.670",
    "Trust | 0.677",
    "Convenience | 0.534",
    "Churn risk | 1.9%",
    "Quadrant | Lovemark | model t112-sim",
  ]);
  expect(screen.getByText(/^Latest order/).textContent).toBe(`Latest order ${ORDER_2}, placed 2026-03-01 19:01.`);
  expect(screen.getByText(/^3 scorings of 2 orders\./)).toBeTruthy();
});

test("the latest scoring's fit splits exactly from as expected, input by input", async () => {
  await open(PROFILE);
  expect(screen.getByText(/^Scoring 3 of 3: 2026-03-01 19:41, order/)).toBeTruthy();
  expect(sum()).toBe("Fit 0.605 = 0.743 as expected −0.138 from its gaps");
  expect(screen.getByText("Fit's top drivers here: Speed vs promise, Food & selection, Fee load.")).toBeTruthy();
  expect(bars()).toEqual([
    { input: "Speed vs promise", amount: "−0.114", bar: "bar left worse" },
    { input: "App ease", amount: "+0.012", bar: "bar right better" },
    { input: "Fee load", amount: "+0.042", bar: "bar right better" },
    { input: "Order fidelity", amount: "0.000", bar: null },
    { input: "Courier conduct", amount: "−0.019", bar: "bar left worse" },
    { input: "Food & selection", amount: "−0.059", bar: "bar left worse" },
    { input: "Personalization & delight", amount: "0.000", bar: null },
    { input: "Recovery handling", amount: "0.000", bar: null },
  ]);
});

test("axe finds nothing on a profile, in either view (T409)", async () => {
  await open(PROFILE);
  expect(await violations()).toEqual([]);
  fireEvent.click(screen.getByLabelText("What changed since the scoring before"));
  expect(await violations()).toEqual([]);
});

test("churn risk splits in percentage points, and a part that raises it is worse", async () => {
  await open(PROFILE);
  fireEvent.click(screen.getByRole("button", { name: "Churn risk" }));
  expect(sum()).toBe("Churn risk 1.9% = 1.2% as expected +0.7 pp from its gaps");
  expect(bars()[0]).toEqual({ input: "Speed vs promise", amount: "+0.6 pp", bar: "bar right worse" });
  expect(bars()[2]).toEqual({ input: "Fee load", amount: "−0.2 pp", bar: "bar left better" });
});

test("stepping back through the history; a scoring from before T316 has no split", async () => {
  await open(PROFILE);
  const earlier = screen.getByRole("button", { name: "← Earlier" });
  fireEvent.click(earlier);
  expect(screen.getByText(/^Scoring 2 of 3: 2026-03-01 19:02/)).toBeTruthy();
  expect(sum()).toBe("Fit 0.649 = 0.743 as expected −0.094 from its gaps");
  fireEvent.click(earlier);
  expect(screen.getByText(/^Scoring 1 of 3: 2026-02-20 12:00, order/)).toBeTruthy();
  expect(screen.getByText("This scoring has no split: it was scored before splits were stored (T316).")).toBeTruthy();
  expect((earlier as HTMLButtonElement).disabled).toBe(true);
  fireEvent.click(screen.getByRole("button", { name: "Latest" }));
  expect(screen.getByText(/^Scoring 3 of 3/)).toBeTruthy();
});

test("the change view splits what moved a score since the scoring before", async () => {
  await open(PROFILE);
  fireEvent.click(screen.getByLabelText("What changed since the scoring before"));
  expect(sum()).toBe("Fit 0.649 → 0.605 (−0.044) since 2026-03-01 19:02");
  expect(bars()[0]).toEqual({ input: "Speed vs promise", amount: "−0.044", bar: "bar left worse" });
  expect(bars().slice(1).every((b) => b.amount === "0.000" && b.bar === null)).toBe(true);
  fireEvent.click(screen.getByRole("button", { name: "← Earlier" }));
  expect(
    screen.getByText("One of these two scorings has no split (scored before T316), so the change can't be split."),
  ).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "← Earlier" }));
  expect(screen.getByText("This is the first scoring: there is nothing before it to compare.")).toBeTruthy();
});

test("the history chart follows the chosen score", async () => {
  await open(PROFILE);
  const chart = () => screen.getByRole("img").getAttribute("aria-label");
  expect(chart()).toMatch(/^Fit, 2026-02-20 12:00 to 2026-03-01 19:41: latest 0\.605\./);
  fireEvent.click(screen.getByRole("button", { name: "Love" }));
  expect(chart()).toMatch(/^Love, .*: latest 0\.731\./);
  expect(screen.getByRole("button", { name: "Love" }).getAttribute("aria-pressed")).toBe("true");
});

test("orders are newest first, and the shown scoring's order is marked", async () => {
  await open(PROFILE);
  const orders = document.querySelector("table.orders") as HTMLElement;
  expect(rows(orders)).toEqual([
    "PlacedOrderRestaurantBasketFeesDeliveryCourierIssuesRatings",
    `2026-03-01 19:01${ORDER_2}rest-23฿191.18 (promo)฿10.4411 min latecourier-31—food 4 · courier 4`,
    `2026-02-20 11:30${ORDER_1}rest-23฿191.18฿10.444 min earlycourier-31cold—`,
  ]);
  expect(orders.querySelector("tr.picked")?.textContent).toContain(ORDER_2);
  fireEvent.click(screen.getByRole("button", { name: "← Earlier" }));
  fireEvent.click(screen.getByRole("button", { name: "← Earlier" }));
  expect(orders.querySelector("tr.picked")?.textContent).toContain(ORDER_1);
});

test("no actions yet; the main action is for analysts and up", async () => {
  await open(PROFILE);
  expect(screen.getByText(/^None recorded yet\. Automations start in Phase 5/)).toBeTruthy();
  expect(
    screen.getByText("Only analysts and admins will be able to trigger an action by hand (Phase 5)."),
  ).toBeTruthy();
  cleanup();
  await open(PROFILE, fakeApi(), "analyst");
  expect(screen.getByText("Triggering an action by hand comes with Phase 5.")).toBeTruthy();
});

test("an unknown customer says so", async () => {
  const api = fakeApi();
  api.answers["/customers/nobody"] = new ApiError(404, "No customer 'nobody'.");
  await open("/customers/nobody", api);
  expect(screen.getByRole("alert").textContent).toBe("No customer nobody.");
});

test("while nothing changes, a check is one read of the customer", async () => {
  const api = await open(PROFILE);
  await wait(REFRESH_MS * 3);
  expect(api.asked(PROFILE)).toBe(4);
  expect(api.asked(ALL)).toBe(1);
  expect(api.asked(NEW)).toBe(0);
  expect(api.asked(ORDER_PAGE)).toBe(1);
});

test("a hidden tab is not checked", async () => {
  const api = await open(PROFILE);
  visibility = "hidden";
  await wait(REFRESH_MS * 2);
  expect(api.asked(PROFILE)).toBe(1);
});

test("when the customer changes, the new scorings and orders come in, and the latest is shown", async () => {
  const api = await open(PROFILE);
  const NEWEST = scoring("e4", ORDER_2, "2026-03-01T13:30:00Z", { x1: -0.114, x3: 0.042, x6: -0.059, x8: 0.05 });
  api.answers[PROFILE] = state(NEWEST);
  api.answers[NEW] = { scores: [NEWEST, LATEST, BEFORE], next_cursor: "more" };
  api.answers[ORDER_PAGE] = {
    orders: [{ ...ORDERS_NOW[0]!, ratings: { food: 4, courier: 4, overall: 5 } }, ORDERS_NOW[1]!],
    next_cursor: null,
  };
  await wait(REFRESH_MS);
  expect(api.asked(NEW)).toBe(1); // it stopped at a scoring it already held
  expect(api.asked(ALL)).toBe(1);
  expect(screen.getByText(/^4 scorings of 2 orders\./)).toBeTruthy();
  expect(screen.getByText(/^Scoring 4 of 4: 2026-03-01 20:30/)).toBeTruthy();
  expect(sum()).toBe("Fit 0.662 = 0.743 as expected −0.081 from its gaps");
  expect(document.querySelector("tr.picked")?.textContent).toContain("overall 5");
});

test("a scoring picked from the past stays picked when new ones come in", async () => {
  const api = await open(PROFILE);
  fireEvent.click(screen.getByRole("button", { name: "← Earlier" }));
  const NEWEST = scoring("e4", ORDER_2, "2026-03-01T13:30:00Z", { x8: 0.05 });
  api.answers[PROFILE] = state(NEWEST);
  api.answers[NEW] = { scores: [NEWEST, LATEST], next_cursor: "more" };
  await wait(REFRESH_MS);
  expect(screen.getByText(/^Scoring 2 of 4: 2026-03-01 19:02/)).toBeTruthy();
});

test("a failed check keeps the profile and says so", async () => {
  const api = await open(PROFILE);
  api.answers[PROFILE] = new ApiError(503, "Firestore is busy");
  await wait(REFRESH_MS);
  expect(screen.getByRole("alert").textContent).toMatch(/^Couldn't check \(Firestore is busy\); this is the profile as of/);
  expect(sum()).toBe("Fit 0.605 = 0.743 as expected −0.138 from its gaps");
});

test("the Overview's riskiest customers link to their profiles", async () => {
  const api = fakeApi();
  api.answers["/overview"] = {
    customers: 2,
    unscored: 1,
    quadrants: { lovemark: 1, brand: 0, fad: 0, product: 0 },
    average_fit: 0.605,
    average_churn_risk: 0.016,
    at_risk: { threshold: 0.1, customers: 0 },
  };
  api.answers["/overview/trend"] = { from: "2026-10-21", to: "2026-11-17", days: [] };
  api.answers["/customers?limit=10"] = api.answers["/customers?limit=50"];
  await open("/overview", api);
  const table = screen.getByRole("heading", { name: "Riskiest customers" }).nextElementSibling as HTMLElement;
  expect(within(table).getByRole("link", { name: ID }).getAttribute("href")).toBe(PROFILE);
});
