// A customer's profile (T404): their scores now, every scoring over time, the
// exact split of the gaps that moved a score (T115, T316), what changed since
// the scoring before, their orders, and the actions they received. While the
// tab is visible it checks the customer every 30 seconds, and reads the new
// scorings and orders only when that changed.
import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router";

import { ApiError, type Api } from "./api";
import { Bars } from "./Bars";
import {
  baht,
  changeOf,
  customerPath,
  delivery,
  INPUT_NAMES,
  mergeScorings,
  signed,
  splitOf,
  SCORE_NAMES,
  SCORES,
  valueOf,
  when,
  type CustomerState,
  type Input,
  type Order,
  type OrderPage,
  type Score,
  type Scoring,
  type ScoringPage,
} from "./customers";
import { LineChart } from "./LineChart";
import { clock, decimal, percent, QUADRANT_NAMES } from "./overview";
import { messageOf, Tile, useWhileVisible } from "./parts";
import { atLeast, whoMay } from "./roles";
import { useStaff } from "./staff";

export const REFRESH_MS = 30_000;
/** Scorings per page when reading the whole history (the API's most). */
export const HISTORY_PAGE = 200;
/** At most this many pages of history: 1,000 scorings. */
export const HISTORY_PAGES = 5;
/** Scorings per page when looking for new ones. */
export const NEW_PAGE = 20;
export const ORDERS = 20;

// The smallest change that fills the history chart's height, and the
// smallest share that fills a bar: smaller ones draw small (as T402's trend).
const SPAN: Record<Score, number> = {
  fit: 0.05,
  love: 0.05,
  respect: 0.05,
  trust: 0.05,
  convenience: 0.05,
  churn_risk: 0.01,
};

function scoringsPath(customerId: string, limit: number, cursor: string | null): string {
  const query = new URLSearchParams({ limit: String(limit) });
  if (cursor !== null) query.set("cursor", cursor);
  return customerPath(customerId, `/scores?${query}`);
}

function ordersPath(customerId: string, cursor: string | null): string {
  const query = new URLSearchParams({ limit: String(ORDERS) });
  if (cursor !== null) query.set("cursor", cursor);
  return customerPath(customerId, `/orders?${query}`);
}

/** The whole history, newest first, up to HISTORY_PAGES pages. */
async function allScorings(
  api: Api,
  id: string,
): Promise<{ scorings: Scoring[]; partial: boolean }> {
  const scorings: Scoring[] = [];
  let cursor: string | null = null;
  for (let pages = 0; pages < HISTORY_PAGES; pages++) {
    const page: ScoringPage = await api.get<ScoringPage>(scoringsPath(id, HISTORY_PAGE, cursor));
    scorings.push(...page.scores);
    cursor = page.next_cursor;
    if (cursor === null) break;
  }
  return { scorings, partial: cursor !== null };
}

/** The scorings held plus the newer ones: pages until one we already hold. */
async function withNewScorings(api: Api, id: string, held: Scoring[]): Promise<Scoring[]> {
  const known = new Set(held.map((s) => s.event_id));
  const fresh: Scoring[] = [];
  let cursor: string | null = null;
  do {
    const page: ScoringPage = await api.get<ScoringPage>(scoringsPath(id, NEW_PAGE, cursor));
    fresh.push(...page.scores);
    if (page.scores.some((s) => known.has(s.event_id))) break;
    cursor = page.next_cursor;
  } while (cursor !== null);
  return mergeScorings(held, fresh);
}

/** Orders newest placed first; a fresh copy of one replaces the held one. */
function mergeOrders(held: Order[], fresh: Order[]): Order[] {
  const byId = new Map(held.map((o) => [o.order_id, o]));
  for (const o of fresh) byId.set(o.order_id, o);
  return [...byId.values()].sort((a, b) =>
    a.placed_at === b.placed_at
      ? b.order_id.localeCompare(a.order_id)
      : b.placed_at.localeCompare(a.placed_at),
  );
}

interface Profile {
  customer: CustomerState;
  scorings: Scoring[];
  /** More history than HISTORY_PAGES pages: the oldest is left out. */
  partial: boolean;
  orders: Order[];
  nextOrders: string | null;
  at: Date;
  /** The last check failed: why; what is shown is from at. */
  stale: string | null;
}

type Loading =
  | { state: "loading" }
  | { state: "failed"; message: string }
  | ({ state: "ready" } & Profile);

export function CustomerPage() {
  const { customerId = "" } = useParams();
  const { api } = useStaff();
  const [profile, setProfile] = useState<Loading>({ state: "loading" });
  const held = useRef<Profile | null>(null);

  useEffect(() => {
    held.current = profile.state === "ready" ? profile : null;
  }, [profile]);

  useEffect(() => {
    let current = true;
    setProfile({ state: "loading" });
    (async () => {
      const customer = await api.get<CustomerState>(customerPath(customerId));
      const [history, orders] = await Promise.all([
        allScorings(api, customerId),
        api.get<OrderPage>(ordersPath(customerId, null)),
      ]);
      return { customer, ...history, orders: orders.orders, nextOrders: orders.next_cursor };
    })().then(
      (loaded) => {
        if (current) setProfile({ state: "ready", ...loaded, at: new Date(), stale: null });
      },
      (failure: unknown) => {
        if (!current) return;
        const message =
          failure instanceof ApiError && failure.status === 404
            ? `No customer ${customerId}.`
            : messageOf(failure);
        setProfile({ state: "failed", message });
      },
    );
    return () => {
      current = false;
    };
  }, [api, customerId]);

  // One read while nothing changed; the new scorings and orders when it did.
  const check = useCallback(async () => {
    const before = held.current;
    if (before === null) return;
    try {
      const customer = await api.get<CustomerState>(customerPath(customerId));
      if (JSON.stringify(customer) === JSON.stringify(before.customer)) {
        setProfile((p) => (p.state === "ready" ? { ...p, at: new Date(), stale: null } : p));
        return;
      }
      const [scorings, orders] = await Promise.all([
        withNewScorings(api, customerId, before.scorings),
        api.get<OrderPage>(ordersPath(customerId, null)),
      ]);
      setProfile((p) =>
        p.state === "ready"
          ? {
              ...p,
              customer,
              scorings,
              orders: mergeOrders(p.orders, orders.orders),
              at: new Date(),
              stale: null,
            }
          : p,
      );
    } catch (failure) {
      setProfile((p) => (p.state === "ready" ? { ...p, stale: messageOf(failure) } : p));
    }
  }, [api, customerId]);
  const tick = useCallback(() => void check(), [check]);
  useWhileVisible(tick, REFRESH_MS);

  async function moreOrders() {
    const before = held.current;
    if (before === null || before.nextOrders === null) return;
    try {
      const page = await api.get<OrderPage>(ordersPath(customerId, before.nextOrders));
      setProfile((p) =>
        p.state === "ready"
          ? { ...p, orders: mergeOrders(p.orders, page.orders), nextOrders: page.next_cursor }
          : p,
      );
    } catch (failure) {
      setProfile((p) => (p.state === "ready" ? { ...p, stale: messageOf(failure) } : p));
    }
  }

  return (
    <section className="customer">
      <p>
        <Link to="/customers">← Customers</Link>
      </p>
      <h1>
        Customer <code>{customerId}</code>
      </h1>
      {profile.state === "loading" && <p>Loading…</p>}
      {profile.state === "failed" && <p role="alert">{profile.message}</p>}
      {profile.state === "ready" && (
        <ProfileView profile={profile} moreOrders={() => void moreOrders()} />
      )}
    </section>
  );
}

function ProfileView({ profile, moreOrders }: { profile: Profile; moreOrders: () => void }) {
  const { customer, scorings, orders, at, stale } = profile;
  const byHand = atLeast(useStaff().role, "analyst");
  const [score, setScore] = useState<Score>("fit");
  // The scoring shown: by event id, or null for the latest (which follows new ones).
  const [picked, setPicked] = useState<string | null>(null);
  const [mode, setMode] = useState<"split" | "change">("split");

  const index = Math.max(0, picked === null ? 0 : scorings.findIndex((s) => s.event_id === picked));
  const shown = scorings[index];
  const pick = (i: number) => setPicked(i === 0 ? null : (scorings[i]?.event_id ?? null));
  const orderCount = new Set(scorings.map((s) => s.order_id)).size;

  return (
    <>
      <p className="updated">
        Updated {clock(at)} · checks for new orders every {REFRESH_MS / 1000} seconds
      </p>
      {stale !== null && (
        <p role="alert">
          Couldn't check ({stale}); this is the profile as of {clock(at)}.
        </p>
      )}
      <Now customer={customer} />
      {customer.latest_placed_at !== null && (
        <p>
          Latest order <code>{customer.latest_order_id}</code>, placed{" "}
          {when(customer.latest_placed_at)}.
        </p>
      )}

      <h2>Scores over time</h2>
      {scorings.length === 0 ? (
        <p>Not scored yet: the first order's events will score them.</p>
      ) : (
        <>
          <ScoreTabs score={score} setScore={setScore} />
          <p>
            {scorings.length} scorings of {orderCount} orders
            {profile.partial ? `, the latest ${scorings.length} only` : ""}. Each event about an
            order scores it again.
          </p>
          <LineChart
            title={SCORE_NAMES[score]}
            points={[...scorings]
              .reverse()
              .map((s) => ({ label: when(s.at), value: valueOf(s, score) }))}
            minSpan={SPAN[score]}
            format={(v) => show(score, v)}
          />

          <h2>What moved it</h2>
          <div className="stepper">
            <button
              type="button"
              onClick={() => pick(index + 1)}
              disabled={index === scorings.length - 1}
            >
              ← Earlier
            </button>
            <span>
              Scoring {scorings.length - index} of {scorings.length}: {when(shown!.at)}, order{" "}
              <code>{shown!.order_id}</code>
            </span>
            <button type="button" onClick={() => pick(index - 1)} disabled={index === 0}>
              Later →
            </button>
            <button type="button" onClick={() => pick(0)} disabled={index === 0}>
              Latest
            </button>
          </div>
          <fieldset className="mode">
            <legend>Show</legend>
            <label>
              <input
                type="radio"
                name="mode"
                checked={mode === "split"}
                onChange={() => setMode("split")}
              />
              The split from "as expected"
            </label>
            <label>
              <input
                type="radio"
                name="mode"
                checked={mode === "change"}
                onChange={() => setMode("change")}
              />
              What changed since the scoring before
            </label>
          </fieldset>
          {mode === "split" ? (
            <SplitView scoring={shown!} score={score} />
          ) : (
            <ChangeView before={scorings[index + 1]} after={shown!} score={score} />
          )}
        </>
      )}

      <h2>Orders</h2>
      <Orders orders={orders} picked={shown?.order_id ?? null} />
      {profile.nextOrders !== null && (
        <button type="button" onClick={moreOrders}>
          Load more orders
        </button>
      )}

      <h2>Actions received</h2>
      <p>
        None recorded yet. Automations start in Phase 5 (T501); each action a rule sends this
        customer, and each it holds back, will be listed here.
      </p>
      <p className={byHand ? "action" : "action read-only"}>
        {byHand
          ? "Triggering an action by hand comes with Phase 5."
          : `Only ${whoMay("analyst")} will be able to trigger an action by hand (Phase 5).`}
      </p>
    </>
  );
}

/** A score's value as the profile shows it: churn risk in percent. */
function show(score: Score, value: number): string {
  return score === "churn_risk" ? percent(value) : decimal(value);
}

/** A share of a score, signed: churn risk in percentage points. */
function showPart(score: Score, value: number): string {
  return score === "churn_risk" ? `${signed(value * 100, 1)} pp` : signed(value);
}

function Now({ customer }: { customer: CustomerState }) {
  if (customer.scores === null) {
    return <p>Not scored yet.</p>;
  }
  const { scores } = customer;
  return (
    <div className="tiles">
      <Tile label="Fit" value={decimal(scores.fit)} />
      <Tile label="Love" value={decimal(scores.love)} />
      <Tile label="Respect" value={decimal(scores.respect)} />
      <Tile label="Trust" value={decimal(scores.trust)} />
      <Tile label="Convenience" value={decimal(scores.convenience)} />
      <Tile label="Churn risk" value={percent(customer.churn_risk)} />
      <Tile
        label="Quadrant"
        value={customer.quadrant === null ? "—" : QUADRANT_NAMES[customer.quadrant]}
        note={`model ${customer.model_version}`}
      />
    </div>
  );
}

function ScoreTabs({ score, setScore }: { score: Score; setScore: (s: Score) => void }) {
  return (
    <div className="score-tabs" role="group" aria-label="Score">
      {SCORES.map((s) => (
        <button key={s} type="button" aria-pressed={s === score} onClick={() => setScore(s)}>
          {SCORE_NAMES[s]}
        </button>
      ))}
    </div>
  );
}

function SplitView({ scoring, score }: { scoring: Scoring; score: Score }) {
  const split = splitOf(scoring, score);
  if (split === null) {
    return <p>This scoring has no split: it was scored before splits were stored (T316).</p>;
  }
  const moved = split.value - split.expected;
  return (
    <>
      <p className="sum">
        {SCORE_NAMES[score]} {show(score, split.value)} = {show(score, split.expected)} as expected{" "}
        {showPart(score, moved)} from its gaps
      </p>
      {scoring.drivers !== undefined && scoring.drivers.length > 0 && (
        <p>Fit's top drivers here: {scoring.drivers.map((d) => INPUT_NAMES[d]).join(", ")}.</p>
      )}
      <ScoreBars parts={split.parts} score={score} label={`${SCORE_NAMES[score]}: each input's share`} />
    </>
  );
}

function ChangeView({
  before,
  after,
  score,
}: {
  before: Scoring | undefined;
  after: Scoring;
  score: Score;
}) {
  if (before === undefined) {
    return <p>This is the first scoring: there is nothing before it to compare.</p>;
  }
  const change = changeOf(before, after, score);
  if (change === null) {
    return (
      <p>One of these two scorings has no split (scored before T316), so the change can't be split.</p>
    );
  }
  return (
    <>
      <p className="sum">
        {SCORE_NAMES[score]} {show(score, change.from)} → {show(score, change.to)} (
        {showPart(score, change.to - change.from)}) since {when(before.at)}
      </p>
      {!change.sameModel && (
        <p className="note">
          The model changed between them ({before.model_version} → {after.model_version}), so the
          parts don't add up to the change.
        </p>
      )}
      <ScoreBars
        parts={change.parts}
        score={score}
        label={`${SCORE_NAMES[score]}: each input's change`}
      />
    </>
  );
}

/** A score's bars: blue is better for the customer (for churn risk, lower is). */
function ScoreBars({
  parts,
  score,
  label,
}: {
  parts: Record<Input, number>;
  score: Score;
  label: string;
}) {
  return (
    <Bars
      parts={parts}
      label={label}
      floor={SPAN[score]}
      format={(value) => showPart(score, value)}
      better={(value) => (score === "churn_risk" ? value < 0 : value > 0)}
      legend={
        <>
          <span className="key better" /> better for the customer than expected{" "}
          <span className="key worse" /> worse. Bars to the right raise{" "}
          {SCORE_NAMES[score].toLowerCase()}, to the left lower it.
        </>
      }
    />
  );
}

function Orders({ orders, picked }: { orders: Order[]; picked: string | null }) {
  if (orders.length === 0) return <p>No orders yet.</p>;
  return (
    <div className="table-scroll">
      <table className="orders">
        <thead>
          <tr>
            <th>Placed</th>
            <th>Order</th>
            <th>Restaurant</th>
            <th>Basket</th>
            <th>Fees</th>
            <th>Delivery</th>
            <th>Courier</th>
            <th>Issues</th>
            <th>Ratings</th>
          </tr>
        </thead>
        <tbody>
          {orders.map((o) => (
            <tr key={o.order_id} className={o.order_id === picked ? "picked" : undefined}>
              <td>{when(o.placed_at)}</td>
              <td>
                <code>{o.order_id}</code>
              </td>
              <td>{o.restaurant_id}</td>
              <td>
                {baht(o.basket_value)}
                {o.promo_used ? " (promo)" : ""}
              </td>
              <td>{baht(o.fees)}</td>
              <td>{delivery(o)}</td>
              <td>{o.courier_id ?? "—"}</td>
              <td>{o.issues.length === 0 ? "—" : o.issues.map((i) => i.issue_type).join(", ")}</td>
              <td>
                {Object.keys(o.ratings).length === 0
                  ? "—"
                  : Object.entries(o.ratings)
                      .map(([kind, stars]) => `${kind} ${stars}`)
                      .join(" · ")}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
