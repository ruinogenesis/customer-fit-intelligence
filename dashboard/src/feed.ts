// Every customer, live (T403): the love matrix's dots move as orders land.
// The design doc's dashboard reads live state through Firestore listeners;
// staff may read customers directly (firestore.rules, T304). Firestore's SDK
// is large, so it loads only when a screen first watches (firestoreFeed.ts).
import type { FeedCustomer } from "./matrix";

export interface CustomerFeed {
  /** Calls onCustomers with every customer now and after each change, until
   * the returned function is called; onError when the feed stops. */
  watch(onCustomers: (customers: FeedCustomer[]) => void, onError: (message: string) => void): () => void;
}

export const firestoreFeed: CustomerFeed = {
  watch(onCustomers, onError) {
    let stop: (() => void) | null = null;
    let stopped = false;
    import("./firestoreFeed").then(
      ({ watchCustomers }) => {
        if (!stopped) stop = watchCustomers(onCustomers, onError);
      },
      (failure: unknown) => onError(`Couldn't load the live feed (${String(failure)})`),
    );
    return () => {
      stopped = true;
      stop?.();
    };
  },
};
