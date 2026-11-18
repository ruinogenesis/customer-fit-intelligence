// T403: the love matrix: every customer as a dot from the live feed, those at
// risk standing out; a group to list and open; movement between two days.
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { createMemoryRouter } from "react-router";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { ApiError } from "../src/api";
import { App } from "../src/App";
import type { FeedCustomer, Movement } from "../src/matrix";
import { FakeApi, FakeAuth, FakeFeed } from "./fakes";

// 10:00 on 2026-11-19 in Bangkok: yesterday, the latest snapshot, is the 18th.
const NOW = new Date("2026-11-19T03:00:00Z");

function customer(
  id: string,
  respect: number,
  love: number,
  quadrant: FeedCustomer["quadrant"],
  churn: number,
): FeedCustomer {
  return { customer_id: id, scores: { respect, love, fit: 0.6 }, churn_risk: churn, quadrant };
}

const CUSTOMERS: FeedCustomer[] = [
  customer("c-a", 0.8, 0.9, "lovemark", 0.02),
  customer("c-b", 0.7, 0.6, "lovemark", 0.12),
  customer("c-c", 0.8, 0.3, "brand", 0.05),
  customer("c-d", 0.2, 0.8, "fad", 0.25),
  customer("c-e", 0.1, 0.2, "product", 0.15),
  { customer_id: "c-f", scores: null, churn_risk: null, quadrant: null },
];

const row = (lovemark = 0, brand = 0, fad = 0, product = 0) => ({ lovemark, brand, fad, product });

function movement(from: string, to: string): { movement: Movement } {
  return {
    movement: {
      from,
      to,
      moves: {
        lovemark: row(140, 1),
        brand: row(1, 4),
        fad: row(0, 0, 11),
        product: row(0, 0, 0, 10),
        unscored: row(3, 0, 1),
      },
      customers: 171,
    },
  };
}

const WEEK = "/matrix?from=2026-11-11&to=2026-11-18";
const DAY = "/matrix?from=2026-11-17&to=2026-11-18";
const FOUR_WEEKS = "/matrix?from=2026-10-21&to=2026-11-18";

function fakeApi(): FakeApi {
  const api = new FakeApi();
  api.answers = {
    [WEEK]: movement("2026-11-11", "2026-11-18"),
    [DAY]: movement("2026-11-17", "2026-11-18"),
    [FOUR_WEEKS]: new ApiError(422, "No daily snapshot holds customers on 2026-10-21."),
  };
  return api;
}

beforeEach(() => {
  vi.useFakeTimers({ now: NOW });
  vi.spyOn(document, "visibilityState", "get").mockImplementation(() => "visible");
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.restoreAllMocks();
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

async function open(path = "/matrix", api = fakeApi(), feed = new FakeFeed()) {
  const auth = new FakeAuth();
  let router!: ReturnType<typeof createMemoryRouter>;
  render(
    <App
      auth={auth}
      api={api}
      feed={feed}
      createRouter={(routes) => (router = createMemoryRouter(routes, { initialEntries: [path] }))}
    />,
  );
  auth.answer({ email: "viewer@example.com", role: "viewer" });
  await settle();
  return { api, feed, router };
}

function dots(): { id: string; className: string; r: string; transform: string }[] {
  return [...document.querySelectorAll("circle.dot")].map((c) => ({
    id: (c.querySelector("title")?.textContent ?? "").split(":")[0]!,
    className: c.getAttribute("class") ?? "",
    r: c.getAttribute("r") ?? "",
    transform: (c as SVGElement).style.transform,
  }));
}

function picker(): string[] {
  return within(screen.getByRole("group", { name: "Quadrant" }))
    .getAllByRole("button")
    .map((b) => `${b.textContent}${b.getAttribute("aria-pressed") === "true" ? " (picked)" : ""}`);
}

function listed(): string[] {
  const table = document.querySelector("table.customer-table");
  return table === null ? [] : [...table.querySelectorAll("tbody tr")].map((r) => r.textContent ?? "");
}

// --- Now ---

test("every scored customer is a dot; those at risk drawn larger, on top", async () => {
  const { feed } = await open();
  expect(screen.getByText("Loading every customer…")).toBeTruthy();
  feed.send(CUSTOMERS);
  expect(dots()).toEqual([
    { id: "c-a", className: "dot", r: "4", transform: "translate(478.4px, 108.8px)" },
    { id: "c-c", className: "dot", r: "4", transform: "translate(478.4px, 425.6px)" },
    { id: "c-b", className: "dot at-risk", r: "6", transform: "translate(425.6px, 267.2px)" },
    { id: "c-d", className: "dot at-risk", r: "6", transform: "translate(161.6px, 161.6px)" },
    { id: "c-e", className: "dot at-risk", r: "6", transform: "translate(108.8px, 478.4px)" },
  ]);
  expect(screen.getByText(/^Live: 5 customers placed, 1 not scored yet · updated 10:00:00$/)).toBeTruthy();
  expect(document.querySelector(".legend")?.textContent).toBe(
    " customer (2)  at risk: churn risk 10% or more (3). Click a dot to open the customer.",
  );
  expect(screen.getByRole("img").getAttribute("aria-label")).toBe(
    "Love matrix: 5 customers by Respect (across) and Love (up), 3 at risk.",
  );
  expect(picker()).toEqual(["Lovemark 2", "Brand 1", "Fad 1", "Product 1"]);
  expect(screen.getByText("Pick a quadrant or a churn-risk level to list those customers.")).toBeTruthy();
});

test("dots move as customers change, each the same dot gliding", async () => {
  const { feed } = await open();
  feed.send(CUSTOMERS);
  const before = document.querySelector("circle.dot");
  const moved = CUSTOMERS.map((c) => (c.customer_id === "c-a" ? customer("c-a", 0.3, 0.9, "fad", 0.02) : c));
  vi.setSystemTime(new Date("2026-11-19T03:00:30Z"));
  feed.send(moved);
  expect(dots()[0]).toEqual({ id: "c-a", className: "dot", r: "4", transform: "translate(214.4px, 108.8px)" });
  expect(document.querySelector("circle.dot")).toBe(before);
  expect(picker()).toEqual(["Lovemark 1", "Brand 1", "Fad 2", "Product 1"]);
  expect(screen.getByText(/updated 10:00:30$/)).toBeTruthy();
});

test("picking a quadrant lists its customers, riskiest first, and fades the rest", async () => {
  const { feed } = await open();
  feed.send(CUSTOMERS);
  fireEvent.click(screen.getByRole("button", { name: "Lovemark 2" }));
  expect(picker()[0]).toBe("Lovemark 2 (picked)");
  expect(screen.getByText("2 customers in Lovemark, riskiest first.")).toBeTruthy();
  expect(listed()).toEqual(["c-b12.0%0.600Lovemark", "c-a2.0%0.600Lovemark"]);
  expect(dots().filter((d) => d.className.includes("faded")).map((d) => d.id)).toEqual(["c-c", "c-d", "c-e"]);
  fireEvent.click(screen.getByRole("button", { name: "Lovemark 2" }));
  expect(screen.getByText("Pick a quadrant or a churn-risk level to list those customers.")).toBeTruthy();
  expect(dots().some((d) => d.className.includes("faded"))).toBe(false);
});

test("a churn-risk level lists those customers, alone or with a quadrant", async () => {
  const { feed } = await open();
  feed.send(CUSTOMERS);
  fireEvent.change(screen.getByLabelText(/Churn risk/), { target: { value: "0.1" } });
  expect(screen.getByText("3 customers with churn risk 10% or more, riskiest first.")).toBeTruthy();
  expect(listed().map((r) => r.slice(0, 3))).toEqual(["c-d", "c-e", "c-b"]);
  fireEvent.click(screen.getByRole("button", { name: "Fad 1" }));
  expect(screen.getByText("1 customer in Fad with churn risk 10% or more, riskiest first.")).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Brand 1" }));
  expect(screen.getByText("0 customers in Brand with churn risk 10% or more, riskiest first.")).toBeTruthy();
  expect(listed()).toEqual([]);
});

test("a dot or a listed customer opens their profile", async () => {
  const { feed, router } = await open();
  feed.send(CUSTOMERS);
  fireEvent.click(screen.getByRole("button", { name: "Fad 1" }));
  expect(screen.getByRole("link", { name: "c-d" }).getAttribute("href")).toBe("/customers/c-d");
  fireEvent.click(document.querySelectorAll("circle.dot")[3]!);
  expect(router.state.location.pathname).toBe("/customers/c-d");
});

test("when the feed stops, the matrix stays as it was and says so", async () => {
  const { feed } = await open();
  feed.send(CUSTOMERS);
  feed.fail("Missing or insufficient permissions.");
  expect(screen.getByRole("alert").textContent).toBe(
    "Live updates stopped (Missing or insufficient permissions.); this is the matrix as of 10:00:00.",
  );
  expect(dots()).toHaveLength(5);
});

test("a feed that fails before any customers says why", async () => {
  const { feed } = await open();
  feed.fail("Missing or insufficient permissions.");
  expect(screen.getByRole("alert").textContent).toBe("No customers: Missing or insufficient permissions.");
});

test("leaving the matrix stops watching", async () => {
  const { feed, router } = await open();
  expect([feed.started, feed.stopped]).toEqual([1, 0]);
  await act(async () => {
    await router.navigate("/customers");
  });
  await settle(); // the Customers screen is its own chunk
  expect(screen.getByRole("heading", { level: 1 }).textContent).toBe("Customers");
  expect([feed.started, feed.stopped]).toEqual([1, 1]);
});

// --- Movement ---

test("by default, movement over the week to yesterday", async () => {
  const { api } = await open();
  expect(api.asked(WEEK)).toBe(1);
  expect(document.querySelector(".matrix .sum")?.textContent).toBe(
    "171 customers scored by the end of 2026-11-18: 165 stayed in their quadrant, 2 moved to another, 4 scored for the first time.",
  );
  const table = document.querySelector("table.movement") as HTMLElement;
  expect(within(table).getAllByRole("row").map((r) => r.textContent)).toEqual([
    "On 2026-11-11 ↓ · on 2026-11-18 →LovemarkBrandFadProduct",
    "Lovemark140100",
    "Brand1400",
    "Fad00110",
    "Product00010",
    "Not scored yet3010",
  ]);
  expect([...table.querySelectorAll("td.moved")].map((c) => c.textContent)).toEqual(["1", "1", "3", "1"]);
  expect(screen.getByRole("button", { name: "A week" }).getAttribute("aria-pressed")).toBe("true");
});

test("the quick spans ask for their days; a day without a snapshot says so", async () => {
  const { api } = await open();
  fireEvent.click(screen.getByRole("button", { name: "Day before" }));
  await settle();
  expect(api.asked(DAY)).toBe(1);
  expect(document.querySelector("table.movement thead")?.textContent).toContain("On 2026-11-17");
  fireEvent.click(screen.getByRole("button", { name: "Four weeks" }));
  await settle();
  expect(screen.getByRole("alert").textContent).toBe("No daily snapshot holds customers on 2026-10-21.");
});

test("a first day that isn't before the second asks nothing", async () => {
  const { api } = await open();
  fireEvent.change(screen.getByLabelText("From the end of"), { target: { value: "2026-11-18" } });
  await settle();
  expect(screen.getByRole("alert").textContent).toBe("Pick a first day before the second.");
  expect(api.calls.filter((c) => c.startsWith("GET /matrix"))).toEqual([`GET ${WEEK}`]);
});
