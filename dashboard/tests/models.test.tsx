// T407: the Models screen: every version for all staff; admins promote and
// roll back after confirming; model cards rendered from markdown, safely.
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { createMemoryRouter } from "react-router";
import { afterEach, expect, test } from "vitest";

import { ApiError } from "../src/api";
import { App } from "../src/App";
import type { Role } from "../src/roles";
import { FakeApi, FakeAuth } from "./fakes";

afterEach(cleanup);

async function modelsAs(role: Role, api = new FakeApi()): Promise<FakeApi> {
  const auth = new FakeAuth();
  render(
    <App
      auth={auth}
      api={api}
      createRouter={(routes) => createMemoryRouter(routes, { initialEntries: ["/models"] })}
    />,
  );
  auth.answer({ email: `${role}@example.com`, role });
  await screen.findByRole("table");
  return api;
}

function row(version: string): HTMLElement {
  const cell = screen.getByText(version, { selector: "td > code" });
  return cell.closest("tr")!;
}

test("all staff see every version, newest first, with status and metrics", async () => {
  await modelsAs("viewer");
  const rows = screen.getAllByRole("row").slice(1);
  expect(rows.map((r) => within(r).getAllByRole("cell")[0]?.textContent)).toEqual([
    "t310-check",
    "t112-sim",
    "t100-old",
  ]);
  const headers = screen.getAllByRole("columnheader").map((h) => h.textContent);
  expect(headers).toEqual(["Version", "Status", "churn_auc", "reorder_auc", "Trained", "Promoted", "Actions"]);
  expect(within(row("t112-sim")).getByText("live")).toBeTruthy();
  expect(within(row("t112-sim")).getByText("0.789")).toBeTruthy();
  expect(within(row("t310-check")).getByText("—")).toBeTruthy(); // never promoted
});

test("only admins get promote buttons", async () => {
  await modelsAs("analyst");
  expect(screen.getByText(/Only admins can promote or roll back\./)).toBeTruthy();
  expect(screen.queryByRole("button", { name: /^Promote|^Roll back/ })).toBeNull();
});

test("an admin promotes a candidate after confirming, and it shows live", async () => {
  const api = await modelsAs("admin");
  expect(screen.queryByRole("button", { name: "Promote t112-sim" })).toBeNull(); // live
  fireEvent.click(screen.getByRole("button", { name: "Promote t310-check" }));
  expect(api.calls).not.toContain("POST /models/t310-check/promote");
  const confirm = screen.getByRole("button", { name: "Make t310-check live" }).closest("td")!;
  expect(confirm.textContent).toBe(
    "Make t310-check live? Every event from then on, on every instance, is scored by it; " +
      "t112-sim is retired.Make t310-check liveCancel",
  );
  fireEvent.click(screen.getByRole("button", { name: "Make t310-check live" }));
  expect((await screen.findByRole("status")).textContent).toBe(
    "t310-check is live. The next event on every instance is scored by it.",
  );
  expect(api.calls).toEqual(["GET /models", "POST /models/t310-check/promote", "GET /models"]);
  expect(within(row("t310-check")).getByText("live")).toBeTruthy();
  expect(within(row("t112-sim")).getByText("retired")).toBeTruthy();
});

test("cancelling promotes nothing", async () => {
  const api = await modelsAs("admin");
  fireEvent.click(screen.getByRole("button", { name: "Promote t310-check" }));
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(screen.queryByRole("button", { name: "Make t310-check live" })).toBeNull();
  expect(api.calls).toEqual(["GET /models"]);
});

test("rolling back is promoting a retired version", async () => {
  await modelsAs("admin");
  fireEvent.click(screen.getByRole("button", { name: "Roll back to this t100-old" }));
  fireEvent.click(screen.getByRole("button", { name: "Make t100-old live" }));
  await screen.findByRole("status");
  expect(within(row("t100-old")).getByText("live")).toBeTruthy();
});

test("a refused promote shows the API's reason and changes nothing", async () => {
  const api = await modelsAs("admin");
  fireEvent.click(screen.getByRole("button", { name: "Promote t310-check" }));
  api.failNext = new ApiError(409, "Not promoted: t310-check: its file does not match its SHA-256");
  fireEvent.click(screen.getByRole("button", { name: "Make t310-check live" }));
  expect((await screen.findByRole("alert")).textContent).toBe(
    "Not promoted: t310-check: its file does not match its SHA-256",
  );
  expect(within(row("t112-sim")).getByText("live")).toBeTruthy();
  expect(screen.getByRole("button", { name: "Make t310-check live" })).toBeTruthy();
});

test("a model card is rendered from markdown, with anything unsafe removed", async () => {
  const card = [
    "# Model card t112-sim",
    "",
    "| Metric | Held out |",
    "| --- | --- |",
    "| churn AUC | 0.789 |",
    "",
    '<img src="x" onerror="alert(1)"><script>alert(2)</script>',
  ].join("\n");
  await modelsAs("viewer", new FakeApi(undefined, { "t112-sim": card }));
  fireEvent.click(screen.getByRole("button", { name: "Model card for t112-sim" }));
  expect(await screen.findByRole("heading", { name: "Model card t112-sim" })).toBeTruthy();
  const rendered = document.querySelector<HTMLElement>(".card")!;
  expect(within(rendered).getByRole("cell", { name: "churn AUC" })).toBeTruthy(); // a table
  expect(document.querySelector(".card script")).toBeNull();
  expect(document.querySelector(".card img")?.getAttribute("onerror")).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Model card for t112-sim" }));
  expect(screen.queryByRole("heading", { name: "Model card t112-sim" })).toBeNull();
});

test("a version without a card says so", async () => {
  await modelsAs("viewer");
  fireEvent.click(screen.getByRole("button", { name: "Model card for t100-old" }));
  expect(await screen.findByText("This version has no model card.")).toBeTruthy();
});

test("a registry that cannot be read says why", async () => {
  const api = new FakeApi();
  api.failNext = new ApiError(503, "This API has no model registry (REGISTRY is unset).");
  const auth = new FakeAuth();
  render(
    <App
      auth={auth}
      api={api}
      createRouter={(routes) => createMemoryRouter(routes, { initialEntries: ["/models"] })}
    />,
  );
  auth.answer({ email: "viewer@example.com", role: "viewer" });
  expect((await screen.findByRole("alert")).textContent).toBe(
    "This API has no model registry (REGISTRY is unset).",
  );
});
