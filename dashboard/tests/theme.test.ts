// T409: the theme choice: System (no choice kept), Light or Dark, kept in this
// browser and applied before the first paint by index.html.
import { readFileSync } from "node:fs";
import { afterEach, expect, test } from "vitest";

import { chooseTheme, keptTheme, THEME_KEY } from "../src/theme";

afterEach(() => {
  localStorage.clear();
  delete document.documentElement.dataset.theme;
});

/** A storage that keeps what it's given, or one that refuses everything. */
function storage(refuse = false): Storage {
  const kept = new Map<string, string>();
  const no = () => {
    throw new DOMException("refused", "SecurityError");
  };
  return {
    getItem: (key: string) => (refuse ? no() : (kept.get(key) ?? null)),
    setItem: (key: string, value: string) => (refuse ? no() : void kept.set(key, value)),
    removeItem: (key: string) => (refuse ? no() : void kept.delete(key)),
    clear: () => kept.clear(),
    key: () => null,
    get length() {
      return kept.size;
    },
  };
}

test("with nothing kept, or something unknown, or storage refused, it's System", () => {
  expect(keptTheme(storage())).toBe("system");
  const odd = storage();
  odd.setItem(THEME_KEY, "blue");
  expect(keptTheme(odd)).toBe("system");
  expect(keptTheme(storage(true))).toBe("system");
});

test("Light or Dark shows at once and is kept; System forgets the choice", () => {
  const root = document.createElement("html");
  const kept = storage();
  chooseTheme("dark", root, kept);
  expect(root.dataset.theme).toBe("dark");
  expect(keptTheme(kept)).toBe("dark");
  chooseTheme("light", root, kept);
  expect(root.dataset.theme).toBe("light");
  expect(kept.getItem(THEME_KEY)).toBe("light");
  chooseTheme("system", root, kept);
  expect(root.hasAttribute("data-theme")).toBe(false);
  expect(kept.getItem(THEME_KEY)).toBeNull();
});

test("when storage refuses, the choice still shows for this visit", () => {
  const root = document.createElement("html");
  chooseTheme("dark", root, storage(true));
  expect(root.dataset.theme).toBe("dark");
});

test("by default it uses this page and this browser's storage", () => {
  chooseTheme("dark");
  expect(document.documentElement.dataset.theme).toBe("dark");
  expect(localStorage.getItem(THEME_KEY)).toBe("dark");
  expect(keptTheme()).toBe("dark");
});

/** index.html's inline script, run as the browser would before the app loads. */
function runPageScript(): void {
  const html = readFileSync("index.html", "utf8");
  const script = /<script>([\s\S]*?)<\/script>/.exec(html)?.[1];
  expect(script).toBeDefined();
  new Function(script!)();
}

test("index.html applies a kept Light or Dark before the first paint, with the same key", () => {
  localStorage.setItem(THEME_KEY, "dark");
  runPageScript();
  expect(document.documentElement.dataset.theme).toBe("dark");
  delete document.documentElement.dataset.theme;
  localStorage.setItem(THEME_KEY, "light");
  runPageScript();
  expect(document.documentElement.dataset.theme).toBe("light");
});

test("index.html leaves the system to decide when nothing usable is kept", () => {
  runPageScript();
  expect(document.documentElement.hasAttribute("data-theme")).toBe(false);
  localStorage.setItem(THEME_KEY, "blue");
  runPageScript();
  expect(document.documentElement.hasAttribute("data-theme")).toBe(false);
});
