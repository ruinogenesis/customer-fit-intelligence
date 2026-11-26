// T402: the Overview screen: today's numbers from the API, refreshed every
// 30 seconds while visible; the last 28 days; the riskiest customers.
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { createMemoryRouter } from "react-router";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { ApiError } from "../src/api";
import { App } from "../src/App";
import type { OverviewNow, Trend } from "../src/overview";
import { REFRESH_MS } from "../src/OverviewPage";
import { violations } from "./axe";
import { FakeApi, FakeAuth } from "./fakes";

const NOW: OverviewNow = {
  customers: 170,
  unscored: 10,
  quadrants: { lovemark: 90, brand: 30, fad: 25, product: 15 },
  average_fit: 0.74213,
  average_churn_risk: 0.0412,
  at_risk: { threshold: 0.1, customers: 12 },
};

const TREND: Trend = {
  from: "2026-10-21",
  to: "2026-11-17",
  days: [
    { day: "2026-11-15", customers: 150, average_fit: 0.71, average_churn_risk: 0.05, quadrants: { lovemark: 80, brand: 30, fad: 25, product: 15 } },
    { day: "2026-11-16", customers: 160, average_fit: 0.73, average_churn_risk: 0.045, quadrants: { lovemark: 85, brand: 30, fad: 30, product: 15 } },
    { day: "2026-11-17", customers: 160, average_fit: 0.735, average_churn_risk: 0.043, quadrants: { lovemark: 90, brand: 30, fad: 25, product: 15 } },
  ],
};

const RISKIEST = {
  customers: [
    { customer_id: "sim315-000042", quadrant: "fad", churn_risk: 0.231, scores: { fit: 0.31 } },
    { customer_id: "sim315-000007", quadrant: "product", churn_risk: 0.118, scores: { fit: 0.45 } },
  ],
  next_cursor: null,
};

function fakeApi(): FakeApi {
  const api = new FakeApi();
  api.answers = {
    "/overview": NOW,
    "/overview/trend": TREND,
    "/customers?limit=10": RISKIEST,
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

/** Let the API's promises (and the lazy page) settle. */
async function settle() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0);
  });
}

async function overviewWith(api = fakeApi()) {
  const auth = new FakeAuth();
  const view = render(
    <App
      auth={auth}
      api={api}
      createRouter={(routes) => createMemoryRouter(routes, { initialEntries: ["/overview"] })}
    />,
  );
  auth.answer({ email: "viewer@example.com", role: "viewer" });
  await settle();
  await settle();
  return { api, view };
}

function tile(label: string): string {
  const found = screen.getByText(label, { selector: ".tile .label" }).closest(".tile")!;
  return [...found.children].map((c) => c.textContent).join(" | ");
}

async function wait(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

test("today's numbers come from the API", async () => {
  await overviewWith();
  expect(tile("Customers")).toBe("Customers | 170 | 160 scored");
  expect(tile("Average fit")).toBe("Average fit | 0.742");
  expect(tile("Average churn risk")).toBe("Average churn risk | 4.1%");
  expect(tile("At risk (churn ≥ 10%)")).toBe("At risk (churn ≥ 10%) | 12 | 8% of scored");
  expect(tile("Lovemark")).toBe("Lovemark | 90 | 56% of scored");
  expect(tile("Product")).toBe("Product | 15 | 9% of scored");
  expect(screen.getByText(/^Updated \d\d:\d\d:\d\d\. Refreshes every 30 seconds\.$/)).toBeTruthy();
});

test("axe finds nothing on the overview, with the numbers table open (T409)", async () => {
  await overviewWith();
  for (const details of document.querySelectorAll("details")) details.open = true;
  expect(await violations()).toEqual([]);
});

test("the numbers refresh every 30 seconds", async () => {
  const { api } = await overviewWith();
  expect(api.asked("/overview")).toBe(1);
  api.answers["/overview"] = { ...NOW, customers: 171, at_risk: { threshold: 0.1, customers: 13 } };
  await wait(REFRESH_MS);
  expect(api.asked("/overview")).toBe(2);
  expect(api.asked("/customers?limit=10")).toBe(2);
  expect(tile("Customers")).toBe("Customers | 171 | 161 scored");
  expect(api.asked("/overview/trend")).toBe(1); // daily: read once
});

test("a hidden tab is not refreshed, and is refreshed as soon as it shows again", async () => {
  const { api } = await overviewWith();
  visibility = "hidden";
  await wait(REFRESH_MS * 3);
  expect(api.asked("/overview")).toBe(1);
  visibility = "visible";
  await act(async () => {
    document.dispatchEvent(new Event("visibilitychange"));
    await vi.advanceTimersByTimeAsync(0);
  });
  expect(api.asked("/overview")).toBe(2);
});

test("leaving the screen stops the refreshing", async () => {
  const { api, view } = await overviewWith();
  view.unmount();
  await wait(REFRESH_MS * 2);
  expect(api.asked("/overview")).toBe(1);
});

test("a failed refresh keeps the last numbers and says so", async () => {
  const { api } = await overviewWith();
  api.answers["/overview"] = new ApiError(503, "Firestore is busy");
  await wait(REFRESH_MS);
  expect(screen.getByRole("alert").textContent).toMatch(
    /^Couldn't refresh \(Firestore is busy\); these are the numbers from \d\d:\d\d:\d\d\.$/,
  );
  expect(tile("Customers")).toBe("Customers | 170 | 160 scored");
});

test("the last days are two charts, ending on their latest values, and a table", async () => {
  await overviewWith();
  expect(screen.getByText("2026-10-21 to 2026-11-17, from the daily snapshots written at 02:00 Bangkok.")).toBeTruthy();
  const [fit, churn] = screen.getAllByRole("img");
  expect(fit!.getAttribute("aria-label")).toMatch(/^Average fit, 2026-11-15 to 2026-11-17: latest 0\.735\./);
  expect(churn!.getAttribute("aria-label")).toMatch(/^Average churn risk, .*: latest 4\.3%\./);
  const numbers = screen.getByText("Show the numbers").closest("details")!;
  fireEvent.click(screen.getByText("Show the numbers"));
  const rows = within(numbers).getAllByRole("row");
  expect(rows.map((r) => r.textContent)).toEqual([
    "DayCustomersAverage fitAverage churn risk",
    "2026-11-151500.7105.0%",
    "2026-11-161600.7304.5%",
    "2026-11-171600.7354.3%",
  ]);
});

test("a flat trend is a flat line on a readable axis, not magnified noise", async () => {
  // Dev's first 28 days: one average every day, apart from floating-point noise.
  const api = fakeApi();
  api.answers["/overview/trend"] = {
    ...TREND,
    days: TREND.days.map((d, i) => ({
      ...d,
      average_fit: 0.48213457 + i * 2 ** -54,
      average_churn_risk: 0.05531 + i * 2 ** -57,
    })),
  };
  await overviewWith(api);
  const [fit, churn] = screen.getAllByRole("img");
  const labels = (svg: Element) => [...svg.querySelectorAll(".tick text")].map((t) => t.textContent);
  expect(labels(fit!)).toEqual(["0.46", "0.48", "0.50"]);
  expect(labels(churn!)).toEqual(["5.0%", "5.5%", "6.0%"]);
});

test("without the daily snapshots the screen says why and shows the rest", async () => {
  const api = fakeApi();
  api.answers["/overview/trend"] = new ApiError(503, "This API reads no BigQuery history (HISTORY is unset), so it shows no trend.");
  await overviewWith(api);
  expect(screen.getByRole("alert").textContent).toBe(
    "No trend: This API reads no BigQuery history (HISTORY is unset), so it shows no trend.",
  );
  expect(tile("Customers")).toBe("Customers | 170 | 160 scored");
});

test("an empty span says when snapshots come", async () => {
  const api = fakeApi();
  api.answers["/overview/trend"] = { ...TREND, days: [] };
  await overviewWith(api);
  expect(screen.getByText(/^No daily snapshots from 2026-10-21 to 2026-11-17 yet/)).toBeTruthy();
});

test("the riskiest customers are listed riskiest first", async () => {
  await overviewWith();
  const table = screen.getByRole("heading", { name: "Riskiest customers" }).nextElementSibling!;
  expect(within(table as HTMLElement).getAllByRole("row").map((r) => r.textContent)).toEqual([
    "CustomerChurn riskFitQuadrant",
    "sim315-00004223.1%0.310Fad",
    "sim315-00000711.8%0.450Product",
  ]);
});
