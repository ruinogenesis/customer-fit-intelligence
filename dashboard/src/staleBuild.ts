// After a deploy, Hosting serves only the new build's files. A tab opened
// before it still runs the old build, and each screen is a file of its own,
// so the first screen it hasn't opened yet fails to load. Vite reports that
// as vite:preloadError: reload once, to get the new build. Not again within
// a minute, so a screen that truly fails shows its error (App's ScreenError)
// instead of reloading forever.

export const RELOADED_AT = "cfi-reloaded-at";
export const AGAIN_AFTER_MS = 60_000;

type Win = Pick<Window, "addEventListener" | "sessionStorage"> & {
  location: Pick<Location, "reload">;
};

export function reloadOnStaleBuild(win: Win = window, now: () => number = Date.now): void {
  win.addEventListener("vite:preloadError", (event) => {
    let last = 0;
    try {
      last = Number(win.sessionStorage.getItem(RELOADED_AT)) || 0;
    } catch {
      // Storage refused (a private window, say): reload anyway, once.
    }
    if (now() - last < AGAIN_AFTER_MS) return;
    try {
      win.sessionStorage.setItem(RELOADED_AT, String(now()));
    } catch {
      return; // Without storage there is no "once": let the error show.
    }
    event.preventDefault();
    win.location.reload();
  });
}
