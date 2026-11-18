// The customer feed on Firestore (feed.ts): one listener on customers, as the
// signed-in staff member (firebase.ts set up the app and its sign-in). A read
// per customer when it starts, then one per customer that changes.
import { getApp } from "firebase/app";
import {
  collection,
  connectFirestoreEmulator,
  getFirestore,
  onSnapshot,
  type DocumentData,
} from "firebase/firestore";

import type { FeedCustomer } from "./matrix";

/** npm run dev: the emulators (as firebase.ts). */
const EMULATORS = import.meta.env.DEV && import.meta.env.MODE !== "cloud";
const FIRESTORE_EMULATOR = { host: "127.0.0.1", port: 8080 };
let connected = false;

function customer(id: string, data: DocumentData): FeedCustomer {
  const scores = data.scores ?? null;
  return {
    customer_id: id,
    scores:
      scores === null
        ? null
        : { love: scores.love, respect: scores.respect, fit: scores.fit },
    churn_risk: data.churn_risk ?? null,
    quadrant: data.quadrant ?? null,
  };
}

export function watchCustomers(
  onCustomers: (customers: FeedCustomer[]) => void,
  onError: (message: string) => void,
): () => void {
  const db = getFirestore(getApp());
  if (EMULATORS && !connected) {
    connectFirestoreEmulator(db, FIRESTORE_EMULATOR.host, FIRESTORE_EMULATOR.port);
    connected = true;
  }
  return onSnapshot(
    collection(db, "customers"),
    (snapshot) => onCustomers(snapshot.docs.map((doc) => customer(doc.id, doc.data()))),
    (error) => onError(error.message),
  );
}
