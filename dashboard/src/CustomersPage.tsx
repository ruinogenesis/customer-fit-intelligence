// The Customers screen (T404): every customer, riskiest first, a page at a
// time, filtered by quadrant; each opens their profile.
import { useEffect, useState } from "react";

import { QUADRANT_NAMES, QUADRANTS, type ListedCustomer, type Quadrant } from "./overview";
import { CustomerTable, messageOf } from "./parts";
import { useStaff } from "./staff";

export const PAGE = 50;

type Page = { customers: ListedCustomer[]; next_cursor: string | null };

type Listing =
  | { state: "loading" }
  | { state: "failed"; message: string }
  | { state: "ready"; customers: ListedCustomer[]; next: string | null; more: boolean };

function listPath(quadrant: Quadrant | "", cursor: string | null): string {
  const query = new URLSearchParams({ limit: String(PAGE) });
  if (quadrant !== "") query.set("quadrant", quadrant);
  if (cursor !== null) query.set("cursor", cursor);
  return `/customers?${query}`;
}

export function CustomersPage() {
  const { api } = useStaff();
  const [quadrant, setQuadrant] = useState<Quadrant | "">("");
  const [listing, setListing] = useState<Listing>({ state: "loading" });

  useEffect(() => {
    let current = true;
    setListing({ state: "loading" });
    api.get<Page>(listPath(quadrant, null)).then(
      (page) => {
        if (!current) return;
        setListing({ state: "ready", customers: page.customers, next: page.next_cursor, more: false });
      },
      (failure: unknown) => {
        if (current) setListing({ state: "failed", message: messageOf(failure) });
      },
    );
    return () => {
      current = false;
    };
  }, [api, quadrant]);

  async function more() {
    if (listing.state !== "ready" || listing.next === null) return;
    setListing({ ...listing, more: true });
    try {
      const page = await api.get<Page>(listPath(quadrant, listing.next));
      setListing({
        state: "ready",
        customers: [...listing.customers, ...page.customers],
        next: page.next_cursor,
        more: false,
      });
    } catch (failure) {
      setListing({ state: "failed", message: messageOf(failure) });
    }
  }

  return (
    <section className="customers">
      <h1>Customers</h1>
      <p>Riskiest first. Open a customer to see why their scores sit where they do.</p>
      <label className="filter">
        Quadrant{" "}
        <select
          value={quadrant}
          onChange={(event) => setQuadrant(event.target.value as Quadrant | "")}
        >
          <option value="">All</option>
          {QUADRANTS.map((q) => (
            <option key={q} value={q}>
              {QUADRANT_NAMES[q]}
            </option>
          ))}
        </select>
      </label>
      {listing.state === "loading" && <p>Loading…</p>}
      {listing.state === "failed" && <p role="alert">{listing.message}</p>}
      {listing.state === "ready" && (
        <>
          {listing.customers.length === 0 ? (
            <p>No customers{quadrant === "" ? "" : ` in ${QUADRANT_NAMES[quadrant]}`} yet.</p>
          ) : (
            <CustomerTable customers={listing.customers} />
          )}
          {listing.next !== null && (
            <button type="button" onClick={() => void more()} disabled={listing.more}>
              {listing.more ? "Loading…" : "Load more"}
            </button>
          )}
        </>
      )}
    </section>
  );
}
