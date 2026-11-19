// T409: what axe-core finds wrong on the page as rendered, against WCAG 2.2
// A and AA. Colour contrast is left to tests/contrast.test.ts: jsdom draws
// nothing, so axe can't see colours here.
import axe from "axe-core";
import { vi } from "vitest";

export async function violations(root: Element = document.body): Promise<string[]> {
  // axe waits on real timers; a test's fake ones would stall it.
  const faked = vi.isFakeTimers();
  if (faked) vi.useRealTimers();
  try {
    const results = await axe.run(root, {
      runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"] },
      rules: { "color-contrast": { enabled: false } },
    });
    return results.violations.map(
      (v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).join(", ")}`,
    );
  } finally {
    if (faked) vi.useFakeTimers();
  }
}
