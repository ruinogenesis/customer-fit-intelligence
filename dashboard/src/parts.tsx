// Pieces more than one screen uses: a number tile, the four quadrants laid
// out as on the love matrix, a table of customers that links each to their
// profile (T404), refreshing while the tab is visible, and a failure's message.
import { useEffect, type ReactNode } from "react";
import { Link } from "react-router";

import { customerPath } from "./customers";
import { decimal, percent, QUADRANT_NAMES, type ListedCustomer } from "./overview";

export function Tile({
  label,
  value,
  note,
  className,
}: {
  label: string;
  value: string;
  note?: string;
  className?: string;
}) {
  return (
    <div className={className === undefined ? "tile" : `tile ${className}`}>
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {note !== undefined && <div className="note">{note}</div>}
    </div>
  );
}

/** The quadrants where the love matrix draws them: Love up, Respect across.
 * Each child is placed by its quadrant's class name. */
export function QuadrantMap({ children }: { children: ReactNode }) {
  return (
    <div className="quadrant-map">
      <span className="axis love" aria-hidden="true">
        Love →
      </span>
      {children}
      <span className="axis respect" aria-hidden="true">
        Respect →
      </span>
    </div>
  );
}

/** Customers as rows, riskiest first as given; each id opens the profile. */
export function CustomerTable({ customers }: { customers: ListedCustomer[] }) {
  return (
    <div className="table-scroll">
      <table className="customer-table">
        <thead>
          <tr>
            <th>Customer</th>
            <th>Churn risk</th>
            <th>Fit</th>
            <th>Quadrant</th>
          </tr>
        </thead>
        <tbody>
          {customers.map((c) => (
            <tr key={c.customer_id}>
              <td>
                <Link to={customerPath(c.customer_id)}>
                  <code>{c.customer_id}</code>
                </Link>
              </td>
              <td>{percent(c.churn_risk)}</td>
              <td>{decimal(c.scores?.fit ?? null)}</td>
              <td>{c.quadrant === null ? "not scored" : QUADRANT_NAMES[c.quadrant]}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Call refresh every ms while the tab is visible, and as soon as it shows
 * again: a hidden tab costs no reads. */
export function useWhileVisible(refresh: () => void, ms: number): void {
  useEffect(() => {
    const timer = setInterval(() => {
      if (document.visibilityState !== "hidden") refresh();
    }, ms);
    const shown = () => {
      if (document.visibilityState === "visible") refresh();
    };
    document.addEventListener("visibilitychange", shown);
    return () => {
      clearInterval(timer);
      document.removeEventListener("visibilitychange", shown);
    };
  }, [refresh, ms]);
}

export function messageOf(failure: unknown): string {
  return failure instanceof Error ? failure.message : String(failure);
}
