// Firebase Auth behind the app's AuthService (auth.ts).
//
// The config, Web API key included, is never in the repo or the build:
// Hosting serves the project's own at /__/firebase/init.json, so one build
// works for dev, staging and prod. `npm run dev` uses the emulators' demo
// project instead, with the Auth emulator; `npm run dev:cloud` fetches dev's
// config through the dev server's proxy (vite.config.ts), as Hosting would.
import { FirebaseError, initializeApp, type FirebaseOptions } from "firebase/app";
import {
  connectAuthEmulator,
  getAuth,
  onAuthStateChanged,
  signInWithEmailAndPassword,
  signOut,
} from "firebase/auth";

import { SignInError, signInMessage, type AuthService } from "./auth";
import { roleFrom } from "./roles";

const DEMO: FirebaseOptions = {
  apiKey: "demo-key",
  authDomain: "demo-cfi.firebaseapp.com",
  projectId: "demo-cfi",
};
const AUTH_EMULATOR = "http://127.0.0.1:9099";

/** npm run dev: the emulators. Built, or npm run dev:cloud: a real project. */
const EMULATORS = import.meta.env.DEV && import.meta.env.MODE !== "cloud";

async function config(): Promise<FirebaseOptions> {
  if (EMULATORS) return DEMO;
  const answer = await fetch("/__/firebase/init.json");
  if (!answer.ok) throw new Error(`No Firebase config from Hosting (${answer.status})`);
  return (await answer.json()) as FirebaseOptions;
}

export async function firebaseAuth(): Promise<AuthService> {
  const auth = getAuth(initializeApp(await config()));
  if (EMULATORS) connectAuthEmulator(auth, AUTH_EMULATOR, { disableWarnings: true });
  return {
    watch: (callback) =>
      onAuthStateChanged(auth, (user) => {
        if (user === null) {
          callback(null);
          return;
        }
        // The role is a custom claim on the ID token (python -m cfi_api.roles).
        void user
          .getIdTokenResult()
          .then((token) => callback({ email: user.email ?? "", role: roleFrom(token.claims) }));
      }),
    signIn: async (email, password) => {
      try {
        await signInWithEmailAndPassword(auth, email, password);
      } catch (error) {
        const code = error instanceof FirebaseError ? error.code : "unknown";
        throw new SignInError(signInMessage(code));
      }
    },
    signOut: () => signOut(auth),
    token: async () => {
      if (auth.currentUser === null) throw new Error("Nobody is signed in");
      return auth.currentUser.getIdToken();
    },
  };
}
