// T406: the Automations screen: every rule, by name, with its kind, status,
// holdout share and last change, for all staff; read-only until T501.
import { cleanup, render, screen, within } from "@testing-library/react";
import { createMemoryRouter } from "react-router";
import { afterEach, expect, test } from "vitest";

import { ApiError } from "../src/api";
import { App } from "../src/App";
import type { Role } from "../src/roles";
import { violations } from "./axe";
import { FakeApi, FakeAuth } from "./fakes";

afterEach(cleanup);

function automationsAs(role: Role, api = new FakeApi()): FakeApi {
  const auth = new FakeAuth();
  render(
    <App
      auth={auth}
      api={api}
      createRouter={(routes) => createMemoryRouter(routes, { initialEntries: ["/automations"] })}
    />,
  );
  auth.answer({ email: `${role}@example.com`, role });
  return api;
}

function cells(): string[][] {
  return screen
    .getAllByRole("row")
    .slice(1)
    .map((row) => within(row).getAllByRole("cell").map((cell) => cell.textContent ?? ""));
}

test("all staff see every rule by name, with kind, status, holdout and last change", async () => {
  const api = automationsAs("viewer");
  await screen.findByRole("table");
  const headers = screen.getAllByRole("columnheader").map((header) => header.textContent);
  expect(headers).toEqual(["Rule", "Kind", "Status", "Holdout", "Last changed"]);
  expect(cells()).toEqual([
    ["Brand → Lovemark", "Marketing", "Off", "50%", "2026-11-28 12:00"],
    ["Fad protection", "Marketing", "Off", "50%", "2026-11-28 12:00"],
    ["Fee sensitivity", "Marketing", "Off", "50%", "2026-11-28 12:00"],
    ["Lovemark advocacy", "Marketing", "Off", "50%", "2026-11-28 12:00"],
    ["Service recovery", "Service", "Off", "50%", "2026-11-28 12:00"],
    ["Win-back", "Marketing", "Off", "50%", "2026-11-28 12:00"],
  ]);
  expect(api.calls).toEqual(["GET /rules"]);
});

test("each status is named, and explained under the table", async () => {
  const api = new FakeApi();
  api.rules = api.rules.slice(0, 4);
  api.rules[1]!.status = "dry_run";
  api.rules[2]!.status = "test";
  api.rules[3]!.status = "live";
  api.rules[3]!.holdout_share = 0.1;
  automationsAs("analyst", api);
  await screen.findByRole("table");
  expect(cells().map((row) => [row[2], row[3]])).toEqual([
    ["Off", "50%"],
    ["Dry run", "50%"],
    ["Test", "50%"],
    ["Live", "10%"],
  ]);
  const explained = document.querySelectorAll(".statuses li");
  expect([...explained].map((item) => item.textContent)).toEqual([
    "Off: does nothing; this is the rule's kill switch.",
    "Dry run: logs what it would do, and sends nothing.",
    "Test: acts, and holds back its holdout share to measure the effect.",
    "Live: acts, and keeps a holdout so its uplift stays measurable.",
    "Service rules skip the marketing consent check and the frequency cap.",
  ]);
});

test("with no rules yet, it says how they get added", async () => {
  const api = new FakeApi();
  api.rules = [];
  automationsAs("viewer", api);
  expect((await screen.findByText(/^No rules yet\./)).textContent).toBe(
    "No rules yet. An admin adds the six starter rules with python -m cfi_api.rules seed.",
  );
  expect(screen.queryByRole("table")).toBeNull();
});

test("a refused read shows the API's reason", async () => {
  const api = new FakeApi();
  api.failNext = new ApiError(503, "Firestore is busy.");
  automationsAs("viewer", api);
  expect((await screen.findByRole("alert")).textContent).toBe("Firestore is busy.");
});

test("it says turning rules on or off comes with T501, and who will be able to", async () => {
  automationsAs("admin");
  await screen.findByRole("table");
  expect(
    screen.getByText(
      "Turning rules on or off, and promoting one from test to live, comes with the rules engine (T501).",
    ),
  ).toBeTruthy();
  cleanup();
  automationsAs("analyst");
  await screen.findByRole("table");
  expect(
    screen.getByText(
      "Only admins will be able to turn rules on or off, or promote one from test to live (T501).",
    ).className,
  ).toBe("action read-only");
});

test("axe finds nothing on the rules", async () => {
  automationsAs("viewer");
  await screen.findByRole("table");
  expect(await violations()).toEqual([]);
});
