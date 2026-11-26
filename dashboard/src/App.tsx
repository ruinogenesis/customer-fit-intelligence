// The staff dashboard's shell (T401): sign-in, or a guest view that reads as a
// viewer (T411), then every screen in a nav, each saying what this person's
// role lets them do there.
import { lazy, Suspense, useEffect, useState, type ComponentType, type FormEvent } from "react";
import {
  createBrowserRouter,
  Navigate,
  NavLink,
  Outlet,
  RouterProvider,
  useRouteError,
  type RouteObject,
} from "react-router";

import { httpApi, type Api } from "./api";
import { SignInError, type AuthService, type Session } from "./auth";
import { firestoreFeed, type CustomerFeed } from "./feed";
import { QUADRANT_MEANING, QUADRANT_NAMES, QUADRANTS } from "./overview";
import { QuadrantMap } from "./parts";
import { atLeast, SCREENS, whoMay, type Screen } from "./roles";
import { StaffContext, useStaff } from "./staff";
import { chooseTheme, keptTheme, THEME_NAMES, THEMES, type Theme } from "./theme";

type Router = ReturnType<typeof createBrowserRouter>;

/** Screens built so far, each loaded when first opened (its own chunk, with
 * its own libraries); the rest show what they will hold (ScreenPage). */
const PAGES: Record<string, ComponentType> = {
  overview: lazy(async () => ({ default: (await import("./OverviewPage")).OverviewPage })),
  matrix: lazy(async () => ({ default: (await import("./MatrixPage")).MatrixPage })),
  customers: lazy(async () => ({ default: (await import("./CustomersPage")).CustomersPage })),
  drivers: lazy(async () => ({ default: (await import("./DriversPage")).DriversPage })),
  models: lazy(async () => ({ default: (await import("./ModelsPage")).ModelsPage })),
};

/** One customer's profile (T404), under the Customers screen. */
const Profile = lazy(async () => ({ default: (await import("./CustomerPage")).CustomerPage }));

function loaded(Page: ComponentType) {
  return (
    <Suspense fallback={<p>Loading…</p>}>
      <Page />
    </Suspense>
  );
}

function page(screen: Screen) {
  const Page = PAGES[screen.path];
  return Page === undefined ? <ScreenPage screen={screen} /> : loaded(Page);
}

export const routes: RouteObject[] = [
  {
    path: "/",
    element: <Shell />,
    children: [
      {
        // A screen that fails shows why inside the shell, with the nav kept.
        errorElement: <ScreenError />,
        children: [
          { index: true, element: <Navigate to={SCREENS[0].path} replace /> },
          ...SCREENS.map((screen) => ({ path: screen.path, element: page(screen) })),
          { path: "customers/:customerId", element: loaded(Profile) },
          { path: "*", element: <NotFound /> },
        ],
      },
    ],
  },
];

export function App({
  auth,
  api,
  feed = firestoreFeed,
  createRouter = createBrowserRouter,
}: {
  auth: AuthService;
  /** The API as the signed-in person; by default over HTTP with their token. */
  api?: Api;
  /** Every customer, live; by default from Firestore. */
  feed?: CustomerFeed;
  createRouter?: (routes: RouteObject[]) => Router;
}) {
  // undefined until Firebase says whether someone is signed in.
  const [session, setSession] = useState<Session | null | undefined>(undefined);
  const [router] = useState(() => createRouter(routes));
  const [client] = useState(() => api ?? httpApi(() => auth.token()));
  useEffect(() => auth.watch(setSession), [auth]);

  if (session === undefined) return <p className="startup">Loading…</p>;
  if (session === null) return <SignIn auth={auth} />;
  const signOut = () => void auth.signOut();
  if (session.role === null) return <NoAccess email={session.email} signOut={signOut} />;
  return (
    <StaffContext
      value={{
        email: session.email,
        role: session.role,
        guest: session.guest === true,
        signOut,
        api: client,
        feed,
      }}
    >
      <RouterProvider router={router} />
    </StaffContext>
  );
}

function SignIn({ auth }: { auth: AuthService }) {
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function attempt(signIn: () => Promise<void>) {
    setBusy(true);
    setError(null);
    try {
      await signIn();
    } catch (failure) {
      setError(failure instanceof SignInError ? failure.message : "Couldn't sign in.");
      setBusy(false);
    }
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    void attempt(() => auth.signIn(String(form.get("email")), String(form.get("password"))));
  }

  return (
    <main className="sign-in">
      <section className="intro">
        <h1 className="wordmark">
          <Mark /> Customer Fit Intelligence
        </h1>
        <p className="lede">
          How much each customer loves and respects the service, who is likely to stop ordering,
          and what pulls them down.
        </p>
        <QuadrantMap>
          <dl className="quadrants">
            {QUADRANTS.map((quadrant) => (
              <div key={quadrant} className={quadrant}>
                <dt>{QUADRANT_NAMES[quadrant]}</dt>
                <dd>{QUADRANT_MEANING[quadrant]}</dd>
              </div>
            ))}
          </dl>
        </QuadrantMap>
      </section>
      <section className="panel" aria-labelledby="sign-in-title">
        <h2 id="sign-in-title">Staff sign-in</h2>
        <form onSubmit={submit}>
          <label>
            Email
            <input name="email" type="email" autoComplete="username" required />
          </label>
          <label>
            Password
            <input name="password" type="password" autoComplete="current-password" required />
          </label>
          {error !== null && <p role="alert">{error}</p>}
          <button type="submit" className="primary" disabled={busy}>
            {busy ? "Signing in…" : "Sign in"}
          </button>
        </form>
        <div className="guest">
          <button
            type="button"
            disabled={busy}
            onClick={() => void attempt(() => auth.signInAsGuest())}
          >
            View as guest
          </button>
          <p>See every screen read-only, without an account.</p>
        </div>
      </section>
    </main>
  );
}

/** The product's mark: the four quadrants, Lovemark filled (D5). */
function Mark() {
  return (
    <svg className="mark" viewBox="0 0 20 20" aria-hidden="true">
      <rect x="1" y="1" width="8" height="8" rx="1.5" />
      <rect className="lovemark" x="11" y="1" width="8" height="8" rx="1.5" />
      <rect x="1" y="11" width="8" height="8" rx="1.5" />
      <rect x="11" y="11" width="8" height="8" rx="1.5" />
    </svg>
  );
}

function NoAccess({ email, signOut }: { email: string; signOut: () => void }) {
  return (
    <main className="sign-in">
      <section className="panel">
        <h1>No dashboard access</h1>
        <p>
          You're signed in as {email}, but this account has no dashboard role. Ask an admin to
          grant you one.
        </p>
        <button type="button" onClick={signOut}>
          Sign out
        </button>
      </section>
    </main>
  );
}

function Shell() {
  const { email, role, guest, signOut } = useStaff();
  return (
    <div className="shell">
      <a
        className="skip"
        href="#main"
        onClick={(event) => {
          event.preventDefault();
          document.getElementById("main")?.focus();
        }}
      >
        Skip to content
      </a>
      <header>
        <span className="wordmark">
          <Mark /> Customer Fit Intelligence
        </span>
        <div className="account">
          <span className="who">
            <span className="email">{guest ? "Guest" : email}</span>{" "}
            <span className="role">{role}</span>
          </span>
          <ThemeSwitch />
          <button type="button" onClick={signOut}>
            {guest ? "Exit guest view" : "Sign out"}
          </button>
        </div>
      </header>
      <nav aria-label="Screens">
        {SCREENS.map((screen) => (
          <NavLink key={screen.path} to={screen.path}>
            {screen.title}
          </NavLink>
        ))}
      </nav>
      <main id="main" tabIndex={-1}>
        <Outlet />
      </main>
    </div>
  );
}

/** System, Light or Dark (T409), kept in this browser. */
function ThemeSwitch() {
  const [theme, setTheme] = useState<Theme>(() => keptTheme());
  return (
    <label className="theme">
      Theme{" "}
      <select
        value={theme}
        onChange={(event) => {
          const chosen = event.target.value as Theme;
          chooseTheme(chosen);
          setTheme(chosen);
        }}
      >
        {THEMES.map((option) => (
          <option key={option} value={option}>
            {THEME_NAMES[option]}
          </option>
        ))}
      </select>
    </label>
  );
}

function ScreenPage({ screen }: { screen: Screen }) {
  const { role } = useStaff();
  const allowed = atLeast(role, screen.actionRole);
  return (
    <section className="placeholder">
      <h1>{screen.title}</h1>
      <p className="lede">{screen.shows}.</p>
      <p className={allowed ? "action" : "action read-only"}>
        {allowed
          ? `Main action: ${screen.action}.`
          : `Read-only for you: only ${whoMay(screen.actionRole)} can ${screen.action.toLowerCase()}.`}
      </p>
      <p className="soon">This screen isn't built yet.</p>
    </section>
  );
}

/** A screen that failed to load or to show (staleBuild.ts reloads once first,
 * when a deploy replaced the screen's file). */
function ScreenError() {
  const error = useRouteError();
  return (
    <section>
      <h1>This screen couldn't load</h1>
      <p className="lede">
        If the dashboard was updated while this page was open, reloading gets the new version.
      </p>
      <button type="button" className="primary" onClick={() => window.location.reload()}>
        Reload the page
      </button>
      <details>
        <summary>What went wrong</summary>
        <p className="note">{error instanceof Error ? error.message : String(error)}</p>
      </details>
    </section>
  );
}

function NotFound() {
  return (
    <section>
      <h1>Not found</h1>
      <p>There's no screen here. Pick one from the list.</p>
    </section>
  );
}
