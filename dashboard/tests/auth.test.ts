// T401: sign-in errors in words, and no Firebase config in the source.
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { expect, test } from "vitest";

import { guestMessage, sessionOf, signInMessage } from "../src/auth";

test.each([
  "auth/invalid-credential",
  "auth/invalid-email",
  "auth/user-not-found",
  "auth/wrong-password",
])("%s says the email or password is wrong, not which", (code) => {
  expect(signInMessage(code)).toBe("Wrong email or password.");
});

test("other errors say what went wrong", () => {
  expect(signInMessage("auth/too-many-requests")).toMatch(/Wait a few minutes/);
  expect(signInMessage("auth/network-request-failed")).toMatch(/connection/);
  expect(signInMessage("auth/user-disabled")).toMatch(/disabled/);
  expect(signInMessage("auth/something-new")).toBe("Couldn't sign in (auth/something-new).");
});

test("a guest reads as a viewer; staff by their role claim (T411)", () => {
  expect(sessionOf({ email: null, isAnonymous: true }, {})).toEqual({
    email: "",
    role: "viewer",
    guest: true,
  });
  // A guest's token can't carry a higher role into the app.
  expect(sessionOf({ email: null, isAnonymous: true }, { role: "admin" }).role).toBe("viewer");
  expect(sessionOf({ email: "ana@example.com", isAnonymous: false }, { role: "analyst" })).toEqual({
    email: "ana@example.com",
    role: "analyst",
  });
  expect(sessionOf({ email: "bo@example.com", isAnonymous: false }, {}).role).toBeNull();
});

test("a guest sign-in turned off says so; other errors as for staff", () => {
  expect(guestMessage("auth/operation-not-allowed")).toMatch(/^Guest view is turned off/);
  expect(guestMessage("auth/admin-restricted-operation")).toMatch(/^Guest view is turned off/);
  expect(guestMessage("auth/network-request-failed")).toMatch(/connection/);
});

test("no Google API key is written into the source: Hosting serves the config", () => {
  const source = join(import.meta.dirname, "..", "src");
  for (const name of readdirSync(source)) {
    expect(readFileSync(join(source, name), "utf8")).not.toMatch(/AIza[0-9A-Za-z_-]{35}/);
  }
});
