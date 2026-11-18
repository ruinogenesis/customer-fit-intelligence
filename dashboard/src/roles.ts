// Staff roles (T303) and the dashboard's screens (design doc, T401).

export const ROLES = ["viewer", "analyst", "admin"] as const;
export type Role = (typeof ROLES)[number];

/** The role on a Firebase ID token's claims, or null if none or unknown. */
export function roleFrom(claims: Record<string, unknown>): Role | null {
  const role = claims["role"];
  return typeof role === "string" && (ROLES as readonly string[]).includes(role)
    ? (role as Role)
    : null;
}

/** True if role ranks at least as high as needed: viewer < analyst < admin. */
export function atLeast(role: Role, needed: Role): boolean {
  return ROLES.indexOf(role) >= ROLES.indexOf(needed);
}

/** Who may take an action, in words: the roles from needed upwards. */
export function whoMay(needed: Role): string {
  return { viewer: "all staff", analyst: "analysts and admins", admin: "admins" }[needed];
}

export interface Screen {
  path: string;
  title: string;
  shows: string;
  action: string;
  /** The lowest role that may take the main action; every role reads. */
  actionRole: Role;
  task: string;
}

/** Every screen is readable by all staff (firestore.rules, T304). */
export const SCREENS: readonly Screen[] = [
  {
    path: "overview",
    title: "Overview",
    shows: "Fit trend, quadrant counts, churn risk, rules firing today",
    action: "Spot a change worth digging into",
    actionRole: "viewer",
    task: "T402",
  },
  {
    path: "matrix",
    title: "Love matrix",
    shows: "Every customer as a dot on Love × Respect; movement between two dates",
    action: "Select a group, open a customer",
    actionRole: "viewer",
    task: "T403",
  },
  {
    path: "customers",
    title: "Customers",
    shows: "Scores over time, the gaps that moved them, orders, actions received",
    action: "Trigger an action by hand",
    actionRole: "analyst",
    task: "T404",
  },
  {
    path: "drivers",
    title: "Drivers",
    shows: "Which inputs pull fit down most, by restaurant and courier",
    action: "Hand an operational fix to the right owner",
    actionRole: "viewer",
    task: "T405",
  },
  {
    path: "automations",
    title: "Automations",
    shows: "Rules, their status, caps, and uplift vs holdout",
    action: "Turn rules on or off, promote from test to live",
    actionRole: "admin",
    task: "T406",
  },
  {
    path: "models",
    title: "Models",
    shows: "Versions, metrics and model cards",
    action: "Promote or roll back",
    actionRole: "admin",
    task: "T407",
  },
];
