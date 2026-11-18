// T401: sign-in, then a shell whose screens say what each role may do.
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { createMemoryRouter } from "react-router";
import { afterEach, expect, test } from "vitest";

import { App } from "../src/App";
import type { Role } from "../src/roles";
import { FakeApi, FakeAuth } from "./fakes";

afterEach(cleanup);

function open(auth: FakeAuth, path = "/") {
  render(
    <App
      auth={auth}
      api={new FakeApi()}
      createRouter={(routes) => createMemoryRouter(routes, { initialEntries: [path] })}
    />,
  );
}

function signedInAs(role: Role | null, path = "/"): FakeAuth {
  const auth = new FakeAuth();
  open(auth, path);
  auth.answer({ email: `${role ?? "nobody"}@example.com`, role });
  return auth;
}

function signInWith(email: string, password: string) {
  fireEvent.change(screen.getByLabelText("Email"), { target: { value: email } });
  fireEvent.change(screen.getByLabelText("Password"), { target: { value: password } });
  fireEvent.click(screen.getByRole("button", { name: "Sign in" }));
}

test("until Firebase answers, it says it is loading", () => {
  open(new FakeAuth());
  expect(screen.getByText("Loading…")).toBeTruthy();
});

test("signed out, it asks to sign in, and a wrong password says so", async () => {
  const auth = new FakeAuth({ "ana@example.com": { password: "right", role: "analyst" } });
  open(auth);
  auth.answer(null);
  signInWith("ana@example.com", "wrong");
  expect((await screen.findByRole("alert")).textContent).toBe("Wrong email or password.");
  expect(auth.signedIn).toBeNull();
});

test("signing in opens the overview, with every screen in the nav", async () => {
  const auth = new FakeAuth({ "ana@example.com": { password: "right", role: "analyst" } });
  open(auth);
  auth.answer(null);
  signInWith("ana@example.com", "right");
  expect(await screen.findByRole("heading", { level: 1, name: "Overview" })).toBeTruthy();
  const nav = screen.getByRole("navigation", { name: "Screens" });
  expect(within(nav).getAllByRole("link").map((link) => link.textContent)).toEqual([
    "Overview",
    "Love matrix",
    "Customers",
    "Drivers",
    "Automations",
    "Models",
  ]);
  expect(screen.getByRole("banner").textContent).toContain("ana@example.com analyst");
});

test("a viewer reads the drivers screen; acting on customers is for analysts up", () => {
  signedInAs("viewer", "/drivers");
  expect(screen.getByText("Main action: Hand an operational fix to the right owner.")).toBeTruthy();
  fireEvent.click(screen.getByRole("link", { name: "Customers" }));
  expect(
    screen.getByText(
      "Read-only for you: only analysts and admins can trigger an action by hand.",
    ),
  ).toBeTruthy();
});

test("an analyst may act on customers, but not on automations", () => {
  signedInAs("analyst", "/customers");
  expect(screen.getByText("Main action: Trigger an action by hand.")).toBeTruthy();
  fireEvent.click(screen.getByRole("link", { name: "Automations" }));
  expect(screen.getByText(/Read-only for you: only admins/)).toBeTruthy();
});

test("the models screen is its own page (T407)", async () => {
  signedInAs("admin", "/models");
  expect(await screen.findByRole("heading", { level: 1, name: "Models" })).toBeTruthy();
  expect(await screen.findByRole("button", { name: "Promote t310-check" })).toBeTruthy();
});

test("an account with no role gets no access, and can sign out", () => {
  signedInAs(null);
  expect(screen.getByRole("heading", { name: "No dashboard access" })).toBeTruthy();
  expect(screen.queryByRole("navigation")).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Sign out" }));
  expect(screen.getByRole("button", { name: "Sign in" })).toBeTruthy();
});

test("an unknown address shows not found inside the shell", () => {
  signedInAs("viewer", "/nowhere");
  expect(screen.getByRole("heading", { level: 1, name: "Not found" })).toBeTruthy();
  expect(screen.getByRole("navigation", { name: "Screens" })).toBeTruthy();
});

test("signing out returns to the sign-in form", () => {
  signedInAs("admin");
  fireEvent.click(screen.getByRole("button", { name: "Sign out" }));
  expect(screen.getByRole("button", { name: "Sign in" })).toBeTruthy();
});
