// T401: staff roles, ranked, and the screens every role reads.
import { describe, expect, test } from "vitest";

import { atLeast, roleFrom, ROLES, SCREENS, whoMay } from "../src/roles";

describe("roleFrom", () => {
  test.each(ROLES)("reads the %s claim", (role) => {
    expect(roleFrom({ role })).toBe(role);
  });

  test.each([{}, { role: "owner" }, { role: 1 }, { role: null }])(
    "gives null for %j: no dashboard role",
    (claims) => {
      expect(roleFrom(claims)).toBeNull();
    },
  );
});

test("roles rank viewer < analyst < admin", () => {
  expect(ROLES.filter((role) => atLeast(role, "viewer"))).toEqual(["viewer", "analyst", "admin"]);
  expect(ROLES.filter((role) => atLeast(role, "analyst"))).toEqual(["analyst", "admin"]);
  expect(ROLES.filter((role) => atLeast(role, "admin"))).toEqual(["admin"]);
});

test("whoMay names the roles from the one needed upwards", () => {
  expect(ROLES.map(whoMay)).toEqual(["all staff", "analysts and admins", "admins"]);
});

test("the six screens of the design doc, in order, each at its own path", () => {
  expect(SCREENS.map((screen) => screen.title)).toEqual([
    "Overview",
    "Love matrix",
    "Customers",
    "Drivers",
    "Automations",
    "Models",
  ]);
  expect(new Set(SCREENS.map((screen) => screen.path)).size).toBe(SCREENS.length);
  expect(SCREENS.every((screen) => ROLES.includes(screen.actionRole))).toBe(true);
});
