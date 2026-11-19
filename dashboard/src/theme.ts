// Light or dark (T409). Without a choice the system setting decides, through
// CSS (light-dark() in styles.css); a choice sets data-theme on <html> and is
// kept in this browser. index.html applies a kept choice before the first
// paint, with the same key and values: keep the two in step.

export const THEMES = ["system", "light", "dark"] as const;
export type Theme = (typeof THEMES)[number];

export const THEME_NAMES: Record<Theme, string> = {
  system: "System",
  light: "Light",
  dark: "Dark",
};

export const THEME_KEY = "cfi-theme";

/** The kept choice; "system" when there is none, or storage can't be read. */
export function keptTheme(storage: Pick<Storage, "getItem"> | undefined = safeStorage()): Theme {
  try {
    const kept = storage?.getItem(THEME_KEY);
    return kept === "light" || kept === "dark" ? kept : "system";
  } catch {
    return "system";
  }
}

/** Show a theme now, and keep it (or forget the choice, for "system"). */
export function chooseTheme(
  theme: Theme,
  root: HTMLElement = document.documentElement,
  storage: Pick<Storage, "setItem" | "removeItem"> | undefined = safeStorage(),
): void {
  if (theme === "system") delete root.dataset.theme;
  else root.dataset.theme = theme;
  try {
    if (theme === "system") storage?.removeItem(THEME_KEY);
    else storage?.setItem(THEME_KEY, theme);
  } catch {
    // Storage refused (a private window, say): the choice lasts this visit.
  }
}

function safeStorage(): Storage | undefined {
  try {
    return window.localStorage;
  } catch {
    return undefined;
  }
}
