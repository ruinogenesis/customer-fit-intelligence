// T412: a screen that fails to load says so inside the shell, with the nav
// kept and a way to reload, instead of React Router's bare error page.
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createMemoryRouter } from "react-router";
import { afterEach, expect, test, vi } from "vitest";

import { App } from "../src/App";
import { FakeApi, FakeAuth } from "./fakes";

// The Models screen's file is gone, as for a tab opened before a deploy.
vi.mock("../src/ModelsPage", () => {
  throw new Error("Failed to fetch dynamically imported module: /assets/ModelsPage-old.js");
});

afterEach(cleanup);

test("a screen that can't load offers a reload, inside the shell", async () => {
  const auth = new FakeAuth();
  render(
    <App
      auth={auth}
      api={new FakeApi()}
      createRouter={(routes) => createMemoryRouter(routes, { initialEntries: ["/models"] })}
    />,
  );
  auth.answer({ email: "viewer@example.com", role: "viewer" });
  expect(
    await screen.findByRole("heading", { level: 1, name: "This screen couldn't load" }),
  ).toBeTruthy();
  expect(screen.getByRole("button", { name: "Reload the page" })).toBeTruthy();
  expect(screen.getByRole("navigation", { name: "Screens" })).toBeTruthy();
  // The error itself, for whoever reports it (Vitest wraps the mock's).
  fireEvent.click(screen.getByText("What went wrong"));
  expect(document.querySelector("details .note")?.textContent).not.toBe("");
});
