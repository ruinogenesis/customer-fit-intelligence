// The staff dashboard's shell (T401): sign-in, then every screen in a nav,
// each saying what this person's role lets them do there.
import { lazy, Suspense, useEffect, useState, type ComponentType, type FormEvent } from "react";
import {
  createBrowserRouter,
  Navigate,
  NavLink,
  Outlet,
  RouterProvider,
  type RouteObject,
} from "react-router";

import { httpApi, type Api } from "./api";
import { SignInError, type AuthService, type Session } from "./auth";
import { atLeast, SCREENS, whoMay, type Screen } from "./roles";
import { StaffContext, useStaff } from "./staff";

type Router = ReturnType<typeof createBrowserRouter>;

/** Screens built so far, each loaded when first opened (its own chunk, with
 * its own libraries); the rest show what they will hold (ScreenPage). */
const PAGES: Record<string, ComponentType> = {
  overview: lazy(async () => ({ default: (await import("./OverviewPage")).OverviewPage })),
  models: lazy(async () => ({ default: (await import("./ModelsPage")).ModelsPage })),
};

function page(screen: Screen) {
  const Page = PAGES[screen.path];
  if (Page === undefined) return <ScreenPage screen={screen} />;
  return (
    <Suspense fallback={<p>Loading…</p>}>
      <Page />
    </Suspense>
  );
}

export const routes: RouteObject[] = [
  {
    path: "/",
    element: <Shell />,
    children: [
      { index: true, element: <Navigate to={SCREENS[0].path} replace /> },
      ...SCREENS.map((screen) => ({ path: screen.path, element: page(screen) })),
      { path: "*", element: <NotFound /> },
    ],
  },
];

export function App({
  auth,
  api,
  createRouter = createBrowserRouter,
}: {
  auth: AuthService;
  /** The API as the signed-in person; by default over HTTP with their token. */
  api?: Api;
  createRouter?: (routes: RouteObject[]) => Router;
}) {
  // undefined until Firebase says whether someone is signed in.
  const [session, setSession] = useState<Session | null | undefined>(undefined);
  const [router] = useState(() => createRouter(routes));
  const [client] = useState(() => api ?? httpApi(() => auth.token()));
  useEffect(() => auth.watch(setSession), [auth]);

  if (session === undefined) return <p className="status">Loading…</p>;
  if (session === null) return <SignIn auth={auth} />;
  const signOut = () => void auth.signOut();
  if (session.role === null) return <NoAccess email={session.email} signOut={signOut} />;
  return (
    <StaffContext value={{ email: session.email, role: session.role, signOut, api: client }}>
      <RouterProvider router={router} />
    </StaffContext>
  );
}

function SignIn({ auth }: { auth: AuthService }) {
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    setBusy(true);
    setError(null);
    try {
      await auth.signIn(String(form.get("email")), String(form.get("password")));
    } catch (failure) {
      setError(failure instanceof SignInError ? failure.message : "Couldn't sign in.");
      setBusy(false);
    }
  }

  return (
    <main className="sign-in">
      <h1>Customer Fit Intelligence</h1>
      <form onSubmit={(event) => void submit(event)}>
        <label>
          Email
          <input name="email" type="email" autoComplete="username" required />
        </label>
        <label>
          Password
          <input name="password" type="password" autoComplete="current-password" required />
        </label>
        {error !== null && <p role="alert">{error}</p>}
        <button type="submit" disabled={busy}>
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </main>
  );
}

function NoAccess({ email, signOut }: { email: string; signOut: () => void }) {
  return (
    <main className="sign-in">
      <h1>No dashboard access</h1>
      <p>
        You're signed in as {email}, but this account has no dashboard role. Ask an admin to
        grant you one.
      </p>
      <button type="button" onClick={signOut}>
        Sign out
      </button>
    </main>
  );
}

function Shell() {
  const { email, role, signOut } = useStaff();
  return (
    <div className="shell">
      <header>
        <strong>Customer Fit Intelligence</strong>
        <span className="who">
          {email} <span className="role">{role}</span>
        </span>
        <button type="button" onClick={signOut}>
          Sign out
        </button>
      </header>
      <nav aria-label="Screens">
        {SCREENS.map((screen) => (
          <NavLink key={screen.path} to={screen.path}>
            {screen.title}
          </NavLink>
        ))}
      </nav>
      <main>
        <Outlet />
      </main>
    </div>
  );
}

function ScreenPage({ screen }: { screen: Screen }) {
  const { role } = useStaff();
  const allowed = atLeast(role, screen.actionRole);
  return (
    <section>
      <h1>{screen.title}</h1>
      <p>{screen.shows}.</p>
      <p className={allowed ? "action" : "action read-only"}>
        {allowed
          ? `Main action: ${screen.action}.`
          : `Read-only for you: only ${whoMay(screen.actionRole)} can ${screen.action.toLowerCase()}.`}
      </p>
      <p className="soon">This screen comes with {screen.task}.</p>
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
