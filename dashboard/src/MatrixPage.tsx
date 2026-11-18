// The love matrix (T403): every customer as a dot on Love × Respect, live from
// the customer feed, with those at risk standing out; pick a quadrant or a
// churn-risk level to list those customers; a dot or a listed customer opens
// their profile (T404). Below, how customers moved between quadrants from one
// day to another (GET /matrix, T313).
import { useEffect, useState } from "react";
import { useNavigate } from "react-router";

import { customerPath } from "./customers";
import {
  AT_RISK,
  bangkokDay,
  countByQuadrant,
  LINE,
  MARGIN,
  matches,
  moveTotals,
  NOT_SCORED,
  SIZE,
  along,
  toDot,
  type Dot,
  type FeedCustomer,
  type Movement,
  type Selection,
} from "./matrix";
import { clock, percent, QUADRANT_NAMES, QUADRANTS, whole, type ListedCustomer } from "./overview";
import { CustomerTable, messageOf } from "./parts";
import { useStaff } from "./staff";

/** The churn-risk levels the list can start from. */
export const CHURN_LEVELS = [0, 0.05, AT_RISK, 0.2] as const;

type Live =
  | { state: "loading" }
  | { state: "failed"; message: string }
  | { state: "ready"; customers: FeedCustomer[]; at: Date; stopped: string | null };

export function MatrixPage() {
  const { feed } = useStaff();
  const [live, setLive] = useState<Live>({ state: "loading" });
  const [selection, setSelection] = useState<Selection>({ quadrant: null, minChurn: 0 });

  useEffect(
    () =>
      feed.watch(
        (customers) => setLive({ state: "ready", customers, at: new Date(), stopped: null }),
        (message) =>
          setLive((before) =>
            before.state === "ready" ? { ...before, stopped: message } : { state: "failed", message },
          ),
      ),
    [feed],
  );

  return (
    <section className="matrix">
      <h1>Love matrix</h1>
      {live.state === "loading" && <p>Loading every customer…</p>}
      {live.state === "failed" && <p role="alert">No customers: {live.message}</p>}
      {live.state === "ready" && (
        <Now {...live} selection={selection} setSelection={setSelection} />
      )}
      <h2>Movement between two days</h2>
      <MovementView />
    </section>
  );
}

function Now({
  customers,
  at,
  stopped,
  selection,
  setSelection,
}: {
  customers: FeedCustomer[];
  at: Date;
  stopped: string | null;
  selection: Selection;
  setSelection: (s: Selection) => void;
}) {
  const navigate = useNavigate();
  const dots = customers.map(toDot).filter((d): d is Dot => d !== null);
  const counts = countByQuadrant(dots);
  const atRisk = dots.filter((d) => d.atRisk).length;
  const chosen = selection.quadrant !== null || selection.minChurn > 0;
  const listed = chosen ? dots.filter((d) => matches(d, selection)) : [];
  const byId = new Map(customers.map((c) => [c.customer_id, c]));

  return (
    <>
      <p className="updated">
        Live: {whole(dots.length)} customers placed, {whole(customers.length - dots.length)} not
        scored yet · updated {clock(at)}
      </p>
      {stopped !== null && (
        <p role="alert">
          Live updates stopped ({stopped}); this is the matrix as of {clock(at)}.
        </p>
      )}
      <Plot
        dots={dots}
        selection={chosen ? selection : null}
        atRisk={atRisk}
        open={(id) => void navigate(customerPath(id))}
      />
      <p className="legend">
        <span className="key customer" /> customer ({whole(dots.length - atRisk)}){" "}
        <span className="key at-risk" /> at risk: churn risk {percent(AT_RISK, 0)} or more (
        {whole(atRisk)}). Click a dot to open the customer.
      </p>

      <h2>Pick a group</h2>
      <div className="picker" role="group" aria-label="Quadrant">
        {QUADRANTS.map((q) => (
          <button
            key={q}
            type="button"
            aria-pressed={selection.quadrant === q}
            onClick={() =>
              setSelection({ ...selection, quadrant: selection.quadrant === q ? null : q })
            }
          >
            {QUADRANT_NAMES[q]} <strong>{whole(counts[q])}</strong>
          </button>
        ))}
      </div>
      <label className="filter">
        Churn risk{" "}
        <select
          value={selection.minChurn}
          onChange={(event) => setSelection({ ...selection, minChurn: Number(event.target.value) })}
        >
          {CHURN_LEVELS.map((level) => (
            <option key={level} value={level}>
              {level === 0
                ? "Any"
                : `${percent(level, 0)} or more${level === AT_RISK ? " (at risk)" : ""}`}
            </option>
          ))}
        </select>
      </label>
      {chosen ? (
        <>
          <p>
            {whole(listed.length)} {listed.length === 1 ? "customer" : "customers"}
            {selection.quadrant === null ? "" : ` in ${QUADRANT_NAMES[selection.quadrant]}`}
            {selection.minChurn > 0 ? ` with churn risk ${percent(selection.minChurn, 0)} or more` : ""}
            , riskiest first.
          </p>
          {listed.length > 0 && (
            <CustomerTable
              customers={[...listed]
                .sort((a, b) => b.churn - a.churn || a.id.localeCompare(b.id))
                .map((d) => listedCustomer(byId.get(d.id)!))}
            />
          )}
        </>
      ) : (
        <p>Pick a quadrant or a churn-risk level to list those customers.</p>
      )}
    </>
  );
}

function listedCustomer(c: FeedCustomer): ListedCustomer {
  return {
    customer_id: c.customer_id,
    quadrant: c.quadrant,
    churn_risk: c.churn_risk,
    scores: c.scores === null ? null : { fit: c.scores.fit },
  };
}

/**
 * Respect across, Love up, D5's lines at 0.5. One colour for everyone; those
 * at risk drawn larger in the critical red, on top (dataviz skill: emphasis,
 * named in the legend). Outside the picked group, dots fade. The dots are for
 * the eye and the mouse; the counts and the list carry the same for keyboards
 * and screen readers.
 */
function Plot({
  dots,
  selection,
  atRisk,
  open,
}: {
  dots: Dot[];
  selection: Selection | null;
  atRisk: number;
  open: (id: string) => void;
}) {
  const low = along(0);
  const high = along(1);
  const middle = along(LINE);
  const side = high - low;
  // At risk last, so they sit on top.
  const ordered = [...dots].sort((a, b) => Number(a.atRisk) - Number(b.atRisk));
  return (
    <svg
      className="matrix-plot"
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      role="img"
      aria-label={`Love matrix: ${dots.length} customers by Respect (across) and Love (up), ${atRisk} at risk.`}
    >
      <rect className="plot" x={low} y={low} width={side} height={side} />
      <line className="midline" x1={middle} y1={low} x2={middle} y2={high} />
      <line className="midline" x1={low} y1={middle} x2={high} y2={middle} />
      <text className="quadrant" x={low + 8} y={low + 20}>
        {QUADRANT_NAMES.fad}
      </text>
      <text className="quadrant" x={high - 8} y={low + 20} textAnchor="end">
        {QUADRANT_NAMES.lovemark}
      </text>
      <text className="quadrant" x={low + 8} y={high - 10}>
        {QUADRANT_NAMES.product}
      </text>
      <text className="quadrant" x={high - 8} y={high - 10} textAnchor="end">
        {QUADRANT_NAMES.brand}
      </text>
      {[0, LINE, 1].map((v) => (
        <g key={v} className="tick">
          <text x={along(v)} y={high + 18} textAnchor="middle">
            {v}
          </text>
          <text x={low - 8} y={SIZE - along(v)} dy="0.32em" textAnchor="end">
            {v}
          </text>
        </g>
      ))}
      <text className="axis" x={SIZE / 2} y={SIZE - MARGIN / 4}>
        Respect →
      </text>
      <text className="axis" transform={`translate(${MARGIN / 4} ${SIZE / 2}) rotate(-90)`}>
        Love →
      </text>
      <g className="dots" aria-hidden="true">
        {ordered.map((dot) => (
          <circle
            key={dot.id}
            className={[
              "dot",
              dot.atRisk ? "at-risk" : "",
              selection !== null && !matches(dot, selection) ? "faded" : "",
            ]
              .filter(Boolean)
              .join(" ")}
            r={dot.atRisk ? 6 : 4}
            style={{ transform: `translate(${dot.x}px, ${dot.y}px)` }}
            onClick={() => open(dot.id)}
          >
            <title>
              {dot.id}: {QUADRANT_NAMES[dot.quadrant]}, churn risk {percent(dot.churn)}
            </title>
          </circle>
        ))}
      </g>
    </svg>
  );
}

// --- Movement (GET /matrix?from=&to=) ---

type Moves =
  | { state: "loading" }
  | { state: "failed"; message: string }
  | { state: "ready"; movement: Movement };

/** A day, n days later (or earlier, for a negative n). */
function shift(day: string, n: number): string {
  const d = new Date(`${day}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + n);
  return d.toISOString().slice(0, 10);
}

/** How far back the quick spans reach. */
const SPANS = [
  { days: 1, label: "Day before" },
  { days: 7, label: "A week" },
  { days: 28, label: "Four weeks" },
] as const;

function MovementView() {
  const { api } = useStaff();
  // Daily snapshots end yesterday (Bangkok): the last day to compare.
  const [latest] = useState(() => bangkokDay(new Date(), 1));
  const [to, setTo] = useState(latest);
  const [from, setFrom] = useState(() => shift(latest, -7));
  const [moves, setMoves] = useState<Moves>({ state: "loading" });

  useEffect(() => {
    if (from >= to) {
      setMoves({ state: "failed", message: "Pick a first day before the second." });
      return;
    }
    let current = true;
    setMoves({ state: "loading" });
    api.get<{ movement: Movement }>(`/matrix?${new URLSearchParams({ from, to })}`).then(
      (answer) => {
        if (current) setMoves({ state: "ready", movement: answer.movement });
      },
      (failure: unknown) => {
        if (current) setMoves({ state: "failed", message: messageOf(failure) });
      },
    );
    return () => {
      current = false;
    };
  }, [api, from, to]);

  return (
    <>
      <div className="days">
        <label>
          From the end of{" "}
          <input type="date" value={from} max={latest} onChange={(e) => setFrom(e.target.value)} />
        </label>
        <label>
          to the end of{" "}
          <input type="date" value={to} max={latest} onChange={(e) => setTo(e.target.value)} />
        </label>
        {SPANS.map(({ days, label }) => (
          <button
            key={days}
            type="button"
            aria-pressed={from === shift(to, -days)}
            onClick={() => setFrom(shift(to, -days))}
          >
            {label}
          </button>
        ))}
      </div>
      <p className="note">From the daily snapshots written at 02:00 Bangkok; the latest is yesterday.</p>
      {moves.state === "loading" && <p>Loading…</p>}
      {moves.state === "failed" && <p role="alert">{moves.message}</p>}
      {moves.state === "ready" && <MovementTable movement={moves.movement} />}
    </>
  );
}

function MovementTable({ movement }: { movement: Movement }) {
  const { stayed, moved, scored } = moveTotals(movement);
  const rows = [...QUADRANTS, NOT_SCORED] as const;
  const name = (was: (typeof rows)[number]) =>
    was === NOT_SCORED ? "Not scored yet" : QUADRANT_NAMES[was];
  return (
    <>
      <p className="sum">
        {whole(movement.customers)} customers scored by the end of {movement.to}:{" "}
        {whole(stayed)} stayed in their quadrant, {whole(moved)} moved to another,{" "}
        {whole(scored)} scored for the first time.
      </p>
      <table className="movement">
        <thead>
          <tr>
            <th scope="col">
              On {movement.from} ↓ · on {movement.to} →
            </th>
            {QUADRANTS.map((q) => (
              <th key={q} scope="col">
                {QUADRANT_NAMES[q]}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((was) => (
            <tr key={was}>
              <th scope="row">{name(was)}</th>
              {QUADRANTS.map((now) => {
                const n = movement.moves[was][now];
                const kind = was === now ? "stayed" : n > 0 ? "moved" : undefined;
                return (
                  <td key={now} className={kind}>
                    {whole(n)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
