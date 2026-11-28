// T401: sign-in, then a shell whose screens say what each role may do.
// T409: a theme switch, a skip link, and nothing axe can find.
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { createMemoryRouter } from "react-router";
import { afterEach, expect, test } from "vitest";

import { App } from "../src/App";
import type { Role } from "../src/roles";
import { violations } from "./axe";
import { FakeApi, FakeAuth } from "./fakes";

afterEach(() => {
  cleanup();
  localStorage.clear();
  delete document.documentElement.dataset.theme;
});

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

test("the customers screen is its own page (T404)", async () => {
  signedInAs("viewer", "/customers");
  expect(await screen.findByRole("heading", { level: 1, name: "Customers" })).toBeTruthy();
});

test("the drivers screen is its own page (T405)", async () => {
  signedInAs("viewer", "/drivers");
  expect(await screen.findByRole("heading", { level: 1, name: "Drivers" })).toBeTruthy();
});

test("the automations screen is its own page (T406)", async () => {
  signedInAs("viewer", "/automations");
  expect(await screen.findByRole("heading", { level: 1, name: "Automations" })).toBeTruthy();
  expect(await screen.findByRole("table")).toBeTruthy();
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

// --- T411 ---

test("a guest sees every screen read-only, and can leave", async () => {
  const auth = new FakeAuth();
  open(auth, "/automations");
  auth.answer(null);
  fireEvent.click(screen.getByRole("button", { name: "View as guest" }));
  expect(await screen.findByRole("heading", { level: 1, name: "Automations" })).toBeTruthy();
  expect(auth.signedIn).toEqual({ email: "", role: "viewer", guest: true });
  expect(screen.getByRole("banner").textContent).toContain("Guest viewer");
  expect(await screen.findByText(/^Only admins will be able to turn rules on or off/)).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Exit guest view" }));
  expect(await screen.findByRole("button", { name: "View as guest" })).toBeTruthy();
  expect(auth.signedIn).toBeNull();
});

test("where guest view is turned off, the guest button says so", async () => {
  const auth = new FakeAuth();
  auth.guestRefusal = "auth/admin-restricted-operation";
  open(auth);
  auth.answer(null);
  fireEvent.click(screen.getByRole("button", { name: "View as guest" }));
  expect((await screen.findByRole("alert")).textContent).toBe(
    "Guest view is turned off on this site. Sign in with a staff account.",
  );
  expect(auth.signedIn).toBeNull();
  expect((screen.getByRole("button", { name: "View as guest" }) as HTMLButtonElement).disabled).toBe(
    false,
  );
});

// --- T409 ---

test("the theme switch shows Light or Dark at once and keeps it; System forgets", () => {
  signedInAs("viewer", "/automations");
  const theme = screen.getByLabelText("Theme") as HTMLSelectElement;
  expect(theme.value).toBe("system");
  expect([...theme.options].map((o) => o.textContent)).toEqual(["System", "Light", "Dark"]);
  fireEvent.change(theme, { target: { value: "dark" } });
  expect(document.documentElement.dataset.theme).toBe("dark");
  expect(localStorage.getItem("cfi-theme")).toBe("dark");
  fireEvent.change(theme, { target: { value: "system" } });
  expect(document.documentElement.hasAttribute("data-theme")).toBe(false);
  expect(localStorage.getItem("cfi-theme")).toBeNull();
});

test("the switch starts on the kept choice", () => {
  localStorage.setItem("cfi-theme", "light");
  signedInAs("viewer", "/automations");
  expect((screen.getByLabelText("Theme") as HTMLSelectElement).value).toBe("light");
});

test("Skip to content is the first stop, and moves focus past the header and nav", () => {
  signedInAs("viewer", "/automations");
  const skip = screen.getByRole("link", { name: "Skip to content" });
  const reachable = document.querySelectorAll("a[href], button, select, input, [tabindex='0']");
  expect(reachable[0]).toBe(skip);
  fireEvent.click(skip);
  expect(document.activeElement).toBe(screen.getByRole("main"));
});

test("axe finds nothing on the sign-in form, with an error showing", async () => {
  const auth = new FakeAuth({ "ana@example.com": { password: "right", role: "analyst" } });
  open(auth);
  auth.answer(null);
  signInWith("ana@example.com", "wrong");
  await screen.findByRole("alert");
  expect(await violations()).toEqual([]);
});

test("axe finds nothing on the no-access page or in the shell", async () => {
  signedInAs(null);
  expect(await violations()).toEqual([]);
  cleanup();
  signedInAs("analyst", "/automations");
  expect(await violations()).toEqual([]);
});
