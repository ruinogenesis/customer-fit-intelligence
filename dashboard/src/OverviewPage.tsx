// The Overview screen (T402): today's numbers, refreshed from the API every
// 30 seconds while the tab is visible, the last 28 days through yesterday,
// and the riskiest customers.
import { useCallback, useEffect, useState } from "react";

import { digitsFor, LineChart } from "./LineChart";
import {
  clock,
  decimal,
  percent,
  QUADRANT_NAMES,
  QUADRANTS,
  share,
  whole,
  type ListedCustomer,
  type OverviewNow,
  type Trend,
} from "./overview";
import { useStaff } from "./staff";

export const REFRESH_MS = 30_000;
const RISKIEST = 10;

// The smallest change that fills a trend chart's height: smaller ones, down
// to the floating-point noise in BigQuery's daily averages, draw nearly flat.
const FIT_SPAN = 0.05;
const CHURN_SPAN = 0.01; // one percentage point

type Now =
  | { state: "loading" }
  | { state: "failed"; message: string }
  | {
      state: "ready";
      overview: OverviewNow;
      riskiest: ListedCustomer[];
      at: Date;
      /** The last refresh failed: why; the numbers shown are from at. */
      stale: string | null;
    };

type Daily = { state: "loading" } | { state: "failed"; message: string } | { state: "ready"; trend: Trend };

function messageOf(failure: unknown): string {
  return failure instanceof Error ? failure.message : String(failure);
}

export function OverviewPage() {
  const { api } = useStaff();
  const [now, setNow] = useState<Now>({ state: "loading" });
  const [daily, setDaily] = useState<Daily>({ state: "loading" });

  const refresh = useCallback(async () => {
    try {
      const [overview, listed] = await Promise.all([
        api.get<OverviewNow>("/overview"),
        api.get<{ customers: ListedCustomer[] }>(`/customers?limit=${RISKIEST}`),
      ]);
      setNow({ state: "ready", overview, riskiest: listed.customers, at: new Date(), stale: null });
    } catch (failure) {
      const message = messageOf(failure);
      setNow((before) =>
        before.state === "ready" ? { ...before, stale: message } : { state: "failed", message },
      );
    }
  }, [api]);

  useEffect(() => {
    void refresh();
    // Only while someone can see it: a hidden tab costs no reads.
    const timer = setInterval(() => {
      if (document.visibilityState !== "hidden") void refresh();
    }, REFRESH_MS);
    const shown = () => {
      if (document.visibilityState === "visible") void refresh();
    };
    document.addEventListener("visibilitychange", shown);
    return () => {
      clearInterval(timer);
      document.removeEventListener("visibilitychange", shown);
    };
  }, [refresh]);

  useEffect(() => {
    api.get<Trend>("/overview/trend").then(
      (trend) => setDaily({ state: "ready", trend }),
      (failure: unknown) => setDaily({ state: "failed", message: messageOf(failure) }),
    );
  }, [api]);

  return (
    <section className="overview">
      <h1>Overview</h1>
      {now.state === "loading" && <p>Loading…</p>}
      {now.state === "failed" && <p role="alert">{now.message}</p>}
      {now.state === "ready" && <Today {...now} />}
      <h2>Last 28 days</h2>
      <TrendView daily={daily} />
      {now.state === "ready" && <Riskiest customers={now.riskiest} />}
    </section>
  );
}

function Today({ overview, at, stale }: { overview: OverviewNow; at: Date; stale: string | null }) {
  const scored = overview.customers - overview.unscored;
  return (
    <>
      <p className="updated">
        Updated {clock(at)} · refreshes every {REFRESH_MS / 1000} seconds
      </p>
      {stale !== null && (
        <p role="alert">
          Couldn't refresh ({stale}); these are the numbers from {clock(at)}.
        </p>
      )}
      <div className="tiles">
        <Tile label="Customers" value={whole(overview.customers)} note={`${whole(scored)} scored`} />
        <Tile label="Average fit" value={decimal(overview.average_fit)} />
        <Tile label="Average churn risk" value={percent(overview.average_churn_risk)} />
        <Tile
          label={`At risk (churn ≥ ${percent(overview.at_risk.threshold, 0)})`}
          value={whole(overview.at_risk.customers)}
          note={`${share(overview.at_risk.customers, scored)} of scored`}
        />
      </div>
      <h2>Quadrants now</h2>
      <div className="tiles">
        {QUADRANTS.map((quadrant) => (
          <Tile
            key={quadrant}
            label={QUADRANT_NAMES[quadrant]}
            value={whole(overview.quadrants[quadrant])}
            note={`${share(overview.quadrants[quadrant], scored)} of scored`}
          />
        ))}
      </div>
    </>
  );
}

function Tile({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="tile">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {note !== undefined && <div className="note">{note}</div>}
    </div>
  );
}

function TrendView({ daily }: { daily: Daily }) {
  if (daily.state === "loading") return <p>Loading the daily snapshots…</p>;
  if (daily.state === "failed") return <p role="alert">No trend: {daily.message}</p>;
  const { trend } = daily;
  if (trend.days.length === 0) {
    return (
      <p>
        No daily snapshots from {trend.from} to {trend.to} yet: each day is written at 02:00
        Bangkok the next night.
      </p>
    );
  }
  const fit = trend.days.map((d) => ({ label: d.day, value: d.average_fit }));
  const churn = trend.days.map((d) => ({ label: d.day, value: d.average_churn_risk }));
  return (
    <>
      <p>
        {trend.from} to {trend.to}, from the daily snapshots written at 02:00 Bangkok.
      </p>
      <div className="charts">
        <LineChart
          title="Average fit"
          points={fit}
          minSpan={FIT_SPAN}
          format={(v) => decimal(v)}
          axis={(v, step) => decimal(v, digitsFor(step))}
        />
        <LineChart
          title="Average churn risk"
          points={churn}
          minSpan={CHURN_SPAN}
          format={(v) => percent(v)}
          axis={(v, step) => percent(v, digitsFor(step * 100))}
        />
      </div>
      <details>
        <summary>Show the numbers</summary>
        <table>
          <thead>
            <tr>
              <th>Day</th>
              <th>Customers</th>
              <th>Average fit</th>
              <th>Average churn risk</th>
            </tr>
          </thead>
          <tbody>
            {trend.days.map((d) => (
              <tr key={d.day}>
                <td>{d.day}</td>
                <td>{whole(d.customers)}</td>
                <td>{decimal(d.average_fit)}</td>
                <td>{percent(d.average_churn_risk)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </>
  );
}

function Riskiest({ customers }: { customers: ListedCustomer[] }) {
  return (
    <>
      <h2>Riskiest customers</h2>
      {customers.length === 0 ? (
        <p>No customers yet.</p>
      ) : (
        <table className="riskiest">
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
                  <code>{c.customer_id}</code>
                </td>
                <td>{percent(c.churn_risk)}</td>
                <td>{decimal(c.scores?.fit ?? null)}</td>
                <td>{c.quadrant === null ? "not scored" : QUADRANT_NAMES[c.quadrant]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </>
  );
}
