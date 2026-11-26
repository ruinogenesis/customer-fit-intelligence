// What the app needs from sign-in, apart from Firebase: firebase.ts provides
// it for real, and the tests give a stand-in.
import { roleFrom, type Role } from "./roles";

export interface Session {
  email: string;
  /** null: signed in, but the account has no dashboard role. */
  role: Role | null;
  /** A guest (T411): signed in anonymously, reading as a viewer. */
  guest?: boolean;
}

/** The session of a Firebase user: a guest reads as a viewer, staff by their role claim. */
export function sessionOf(
  user: { email: string | null; isAnonymous: boolean },
  claims: Record<string, unknown>,
): Session {
  if (user.isAnonymous) return { email: "", role: "viewer", guest: true };
  return { email: user.email ?? "", role: roleFrom(claims) };
}

export interface AuthService {
  /** Calls back with the session now and on every change; returns a stop. */
  watch(callback: (session: Session | null) => void): () => void;
  /** Rejects with a SignInError whose message is fit to show. */
  signIn(email: string, password: string): Promise<void>;
  /** Signs in anonymously, as a guest (T411); rejects as signIn does. */
  signInAsGuest(): Promise<void>;
  signOut(): Promise<void>;
  /** The signed-in person's ID token, for the API (refreshed when needed). */
  token(): Promise<string>;
}

export class SignInError extends Error {}

/** A Firebase Auth error code, in words for the sign-in form. */
export function signInMessage(code: string): string {
  switch (code) {
    case "auth/invalid-credential":
    case "auth/invalid-email":
    case "auth/user-not-found":
    case "auth/wrong-password":
      return "Wrong email or password.";
    case "auth/user-disabled":
      return "This account is disabled; ask an admin.";
    case "auth/too-many-requests":
      return "Too many tries. Wait a few minutes, then try again.";
    case "auth/network-request-failed":
      return "Can't reach the sign-in service. Check your connection.";
    default:
      return `Couldn't sign in (${code}).`;
  }
}

/** A Firebase Auth error code from a guest sign-in, in words. */
export function guestMessage(code: string): string {
  switch (code) {
    case "auth/operation-not-allowed":
    case "auth/admin-restricted-operation":
      return "Guest view is turned off on this site. Sign in with a staff account.";
    default:
      return signInMessage(code);
  }
}
