// T401: sign-in, then a shell whose screens say what each role may do.
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { createMemoryRouter } from "react-router";
import { afterEach, expect, test } from "vitest";

import { App } from "../src/App";
import { SignInError, signInMessage, type AuthService, type Session } from "../src/auth";
import type { Role } from "../src/roles";

afterEach(cleanup);

/** Firebase Auth's stand-in: accounts by email, and a session to report. */
class FakeAuth implements AuthService {
  signedIn: Session | null = null;
  private listener: ((session: Session | null) => void) | null = null;
  private accounts: Record<string, { password: string; role: Role | null }>;

  constructor(accounts: Record<string, { password: string; role: Role | null }> = {}) {
    this.accounts = accounts;
  }

  watch(callback: (session: Session | null) => void): () => void {
    this.listener = callback;
    return () => {
      this.listener = null;
    };
  }

  /** What Firebase says when the page loads. */
  answer(session: Session | null): void {
    act(() => this.listener?.(session));
  }

  async signIn(email: string, password: string): Promise<void> {
    const account = this.accounts[email];
    if (account === undefined || account.password !== password) {
      throw new SignInError(signInMessage("auth/invalid-credential"));
    }
    this.signedIn = { email, role: account.role };
    this.listener?.(this.signedIn);
  }

  async signOut(): Promise<void> {
    this.signedIn = null;
    this.listener?.(null);
  }
}

function open(auth: FakeAuth, path = "/") {
  render(
    <App auth={auth} createRouter={(routes) => createMemoryRouter(routes, { initialEntries: [path] })} />,
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

test("a viewer reads the models screen but may not promote", () => {
  signedInAs("viewer", "/models");
  expect(screen.getByRole("heading", { level: 1, name: "Models" })).toBeTruthy();
  expect(
    screen.getByText("Read-only for you: only admins can promote or roll back."),
  ).toBeTruthy();
});

test("an analyst may act on customers, but not promote models", () => {
  signedInAs("analyst", "/customers");
  expect(screen.getByText("Main action: Trigger an action by hand.")).toBeTruthy();
  fireEvent.click(screen.getByRole("link", { name: "Models" }));
  expect(screen.getByText(/Read-only for you: only admins/)).toBeTruthy();
});

test("an admin may promote models", () => {
  signedInAs("admin", "/models");
  expect(screen.getByText("Main action: Promote or roll back.")).toBeTruthy();
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
