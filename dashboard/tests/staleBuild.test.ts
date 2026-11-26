// T412: a tab opened before a deploy reloads once when a screen's file is
// gone, and never again within a minute, so it can't loop.
import { expect, test, vi } from "vitest";

import { AGAIN_AFTER_MS, RELOADED_AT, reloadOnStaleBuild } from "../src/staleBuild";

function standIn(refuse = false) {
  const target = new EventTarget();
  const stored: Record<string, string> = {};
  const reload = vi.fn();
  const sessionStorage = {
    getItem: (key: string) => {
      if (refuse) throw new Error("refused");
      return stored[key] ?? null;
    },
    setItem: (key: string, value: string) => {
      if (refuse) throw new Error("refused");
      stored[key] = value;
    },
  };
  const win = {
    addEventListener: target.addEventListener.bind(target),
    sessionStorage,
    location: { reload },
  } as unknown as Parameters<typeof reloadOnStaleBuild>[0];
  /** A screen's file fails to load; true if the page reloads instead. */
  const fail = () => {
    const event = new Event("vite:preloadError", { cancelable: true });
    target.dispatchEvent(event);
    return event.defaultPrevented;
  };
  return { win, stored, reload, fail };
}

test("a screen's file gone after a deploy reloads the page, once", () => {
  let now = 1_000_000;
  const tab = standIn();
  reloadOnStaleBuild(tab.win, () => now);
  expect(tab.fail()).toBe(true);
  expect(tab.reload).toHaveBeenCalledTimes(1);
  expect(tab.stored[RELOADED_AT]).toBe("1000000");
  // Failing again straight after the reload: the error shows instead.
  now += 5_000;
  expect(tab.fail()).toBe(false);
  expect(tab.reload).toHaveBeenCalledTimes(1);
  // After the next deploy, a minute or more later, it reloads again.
  now += AGAIN_AFTER_MS;
  expect(tab.fail()).toBe(true);
  expect(tab.reload).toHaveBeenCalledTimes(2);
});

test("without session storage it shows the error rather than risk a loop", () => {
  const tab = standIn(true);
  reloadOnStaleBuild(tab.win, () => 1_000_000);
  expect(tab.fail()).toBe(false);
  expect(tab.reload).not.toHaveBeenCalled();
});
