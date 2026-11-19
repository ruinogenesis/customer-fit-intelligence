// T409: every colour is a theme variable in styles.css, and each text and
// mark is readable on the backgrounds it sits on, in both themes (WCAG 2.2:
// 4.5:1 for text, 3:1 for marks and the focus ring). Computed from the
// stylesheet itself: jsdom draws nothing, so axe can't check colours.
import { readFileSync } from "node:fs";
import { expect, test } from "vitest";

// Vitest runs in dashboard/ (npm test), as in CI.
const CSS = readFileSync("src/styles.css", "utf8");
const ROOT = CSS.slice(CSS.indexOf(":root {"), CSS.indexOf("}", CSS.indexOf(":root {")));
const REST = CSS.slice(CSS.indexOf("}", CSS.indexOf(":root {")));

type Mode = "light" | "dark";
const HEX = "#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?";

/** Each variable's colour in each theme. */
function tokens(): Record<string, Record<Mode, string>> {
  const found: Record<string, Record<Mode, string>> = {};
  const pattern = new RegExp(`--([a-z-]+):\\s*(?:light-dark\\((${HEX}),\\s*(${HEX})\\)|(${HEX}));`, "g");
  for (const [, name, light, dark, both] of ROOT.matchAll(pattern)) {
    found[name!] = { light: (light ?? both)!, dark: (dark ?? both)! };
  }
  return found;
}

function luminance(hex: string): number {
  const full = hex.length === 4 ? Array.from(hex.slice(1), (c) => c + c).join("") : hex.slice(1);
  const [r, g, b] = [0, 2, 4].map((i) => {
    const c = parseInt(full.slice(i, i + 2), 16) / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r! + 0.7152 * g! + 0.0722 * b!;
}

function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi! + 0.05) / (lo! + 0.05);
}

/** Text, and where it is drawn. */
const TEXT: Record<string, string[]> = {
  ink: ["page", "surface", "tint", "badge", "pill", "pill-live", "caution"],
  "ink-muted": ["page", "surface", "tint", "pill"],
  link: ["page", "surface", "tint"],
  danger: ["page", "surface"],
  success: ["page", "surface"],
  // A pressed picker button: the page's colours swapped.
  surface: ["ink"],
};

/** Marks (dots, lines, bars, keys, the focus ring), and where they are drawn. */
const MARKS: Record<string, string[]> = {
  series: ["page", "surface"],
  worse: ["page", "surface"],
  critical: ["page", "surface"],
  focus: ["page", "surface", "tint"],
  rule: ["surface", "tint"],
};

/** Hairlines that only separate (dataviz skill: recessive grid and frame),
 * and the matrix's as-expected band behind its dots (T410); nothing is read
 * from them alone, so they have no contrast floor. */
const HAIRLINES = ["line", "grid", "frame", "band"];

test("every colour outside the theme block is one of its variables", () => {
  expect(REST.match(new RegExp(HEX, "g")) ?? []).toEqual([]);
  expect(REST).not.toMatch(/\brgba?\(|\bhsla?\(|LinkText/);
  const used = new Set([...CSS.matchAll(/var\(--([a-z-]+)\)/g)].map((m) => m[1]));
  const declared = Object.keys(tokens());
  expect([...used].filter((name) => !declared.includes(name!))).toEqual([]);
  expect(declared.filter((name) => !used.has(name))).toEqual([]);
});

test("every variable is decided: checked as text, as a mark, or a hairline", () => {
  const decided = new Set([
    ...Object.keys(TEXT),
    ...Object.values(TEXT).flat(),
    ...Object.keys(MARKS),
    ...Object.values(MARKS).flat(),
    ...HAIRLINES,
  ]);
  expect(Object.keys(tokens()).filter((name) => !decided.has(name))).toEqual([]);
});

for (const mode of ["light", "dark"] as const) {
  test(`${mode}: text is at least 4.5:1 on its backgrounds`, () => {
    const t = tokens();
    const low = Object.entries(TEXT).flatMap(([fg, bgs]) =>
      bgs
        .map((bg) => ({ pair: `${fg} on ${bg}`, ratio: contrast(t[fg]![mode], t[bg]![mode]) }))
        .filter(({ ratio }) => ratio < 4.5)
        .map(({ pair, ratio }) => `${pair}: ${ratio.toFixed(2)}`),
    );
    expect(low).toEqual([]);
  });

  test(`${mode}: marks and the focus ring are at least 3:1 on their backgrounds`, () => {
    const t = tokens();
    const low = Object.entries(MARKS).flatMap(([fg, bgs]) =>
      bgs
        .map((bg) => ({ pair: `${fg} on ${bg}`, ratio: contrast(t[fg]![mode], t[bg]![mode]) }))
        .filter(({ ratio }) => ratio < 3)
        .map(({ pair, ratio }) => `${pair}: ${ratio.toFixed(2)}`),
    );
    expect(low).toEqual([]);
  });
}

test("the ratios are WCAG's: black on white is 21, a colour on itself 1", () => {
  expect(contrast("#000", "#fff")).toBeCloseTo(21, 5);
  expect(contrast("#2a78d6", "#2a78d6")).toBe(1);
  expect(contrast("#767676", "#fff")).toBeCloseTo(4.54, 2);
});

test("the dark theme is chosen by the switch or, without one, by the system", () => {
  expect(CSS).toMatch(/:root \{\n {2}color-scheme: light dark;/);
  expect(CSS).toMatch(/:root\[data-theme="light"\] \{\n {2}color-scheme: light;\n\}/);
  expect(CSS).toMatch(/:root\[data-theme="dark"\] \{\n {2}color-scheme: dark;\n\}/);
});
