// Drivers (T405): which inputs pull fit down at each restaurant or courier
// (GET /drivers, T315). First each input's average share of fit over every
// order in the span; then the groups, worst first, each with the inputs that
// are worse there than over every order; a row opens all eight. The grouping,
// span and minimum live in the URL, so a link shows the same list.
import { useEffect, useState } from "react";
import { useSearchParams } from "react-router";

import { Bars } from "./Bars";
import { INPUT_NAMES, signed } from "./customers";
import {
  BY,
  BY_NAMES,
  capital,
  driversPath,
  LIMIT,
  MIN_ORDERS,
  namedDrivers,
  readView,
  SPANS,
  spanDays,
  type DriverGroup,
  type Drivers,
  type DriversView,
} from "./drivers";
import { addDays, bangkokDay } from "./matrix";
import { decimal, whole } from "./overview";
import { messageOf } from "./parts";
import { useStaff } from "./staff";

/** The smallest average share that fills half a bar's width. */
const SHARE_FLOOR = 0.05;

type Answer =
  | { state: "loading" }
  | { state: "failed"; message: string }
  | { state: "ready"; drivers: Drivers };

function orders(n: number): string {
  return `${whole(n)} ${n === 1 ? "order" : "orders"}`;
}

export function DriversPage() {
  const { api } = useStaff();
  const [today] = useState(() => bangkokDay(new Date()));
  const [search, setSearch] = useSearchParams();
  const view = readView(search, today);
  const path = driversPath(view);
  const [answer, setAnswer] = useState<Answer>({ state: "loading" });
  const [open, setOpen] = useState<string | null>(null);

  const change = (parts: Partial<Record<"by" | "from" | "to" | "min_orders", string>>) =>
    setSearch(
      (before) => {
        const next = new URLSearchParams(before);
        for (const [key, value] of Object.entries(parts)) next.set(key, value);
        return next;
      },
      { replace: true },
    );

  const backwards = view.from > view.to;
  useEffect(() => {
    if (backwards) {
      setAnswer({ state: "failed", message: "Pick a first day on or before the last." });
      return;
    }
    let current = true;
    setAnswer({ state: "loading" });
    api.get<Drivers>(path).then(
      (drivers) => {
        if (current) setAnswer({ state: "ready", drivers });
      },
      (failure: unknown) => {
        if (current) setAnswer({ state: "failed", message: messageOf(failure) });
      },
    );
    return () => {
      current = false;
    };
  }, [api, path, backwards]);

  const days = spanDays(view.from, view.to);
  const minimums = [...new Set<number>([...MIN_ORDERS, view.minOrders])].sort((a, b) => a - b);

  return (
    <section className="drivers">
      <h1>Drivers</h1>
      <p className="lede">
        Which inputs pull fit down at each restaurant or courier, compared with every order in
        the span.
      </p>

      <div className="picker" role="group" aria-label="Group by">
        {BY.map((by) => (
          <button
            key={by}
            type="button"
            aria-pressed={view.by === by}
            onClick={() => change({ by })}
          >
            {capital(BY_NAMES[by].many)}
          </button>
        ))}
      </div>
      <div className="days">
        <label>
          Orders placed from{" "}
          <input
            type="date"
            value={view.from}
            max={today}
            onChange={(e) => change({ from: e.target.value })}
          />
        </label>
        <label>
          to{" "}
          <input
            type="date"
            value={view.to}
            max={today}
            onChange={(e) => change({ to: e.target.value })}
          />
        </label>
        {SPANS.map((n) => (
          <button
            key={n}
            type="button"
            aria-pressed={days === n}
            onClick={() => change({ from: addDays(view.to, 1 - n), to: view.to })}
          >
            {n} days
          </button>
        ))}
        <label>
          At least{" "}
          <select
            value={view.minOrders}
            onChange={(e) => change({ min_orders: e.target.value })}
          >
            {minimums.map((n) => (
              <option key={n} value={n}>
                {orders(n)}
              </option>
            ))}
          </select>
        </label>
      </div>
      <p className="note">
        Bangkok days, both included. Each order counts by its latest scoring; the list leaves out
        groups with fewer orders than the minimum.
      </p>

      {answer.state === "loading" && <p>Loading…</p>}
      {answer.state === "failed" && <p role="alert">{answer.message}</p>}
      {answer.state === "ready" && (
        <Ranking
          drivers={answer.drivers}
          view={view}
          open={open}
          toggle={(id) => setOpen((was) => (was === id ? null : id))}
        />
      )}

      <p className="action">
        Handing a fix to its owner isn't built yet. To pass one on, share this page's link: it
        keeps the grouping, the span and the minimum.
      </p>
    </section>
  );
}

function Ranking({
  drivers,
  view,
  open,
  toggle,
}: {
  drivers: Drivers;
  view: DriversView;
  open: string | null;
  toggle: (id: string) => void;
}) {
  const { overall, groups } = drivers;
  const names = BY_NAMES[drivers.by];
  const fit = overall.fit;
  if (overall.orders === 0 || fit === null) {
    return (
      <p>
        No orders were placed from {drivers.from} to {drivers.to}.
      </p>
    );
  }
  return (
    <>
      <p className="sum">
        {orders(overall.orders)} placed from {drivers.from} to {drivers.to}, average fit{" "}
        {decimal(fit)}.
      </p>

      <h2>Every order</h2>
      <Bars
        parts={overall.drag}
        label="Fit over every order: each input's average share"
        floor={SHARE_FLOOR}
        format={(value) => signed(value)}
        better={(value) => value > 0}
        legend={
          <>
            <span className="key better" /> better than expected{" "}
            <span className="key worse" /> worse. Bars to the left lower fit.
          </>
        }
      />

      <h2>Worst first</h2>
      {groups.length === 0 ? (
        <p>
          No {names.one} has at least {orders(view.minOrders)} in this span. Lower
          the minimum to see more.
        </p>
      ) : (
        <>
          <p>
            {whole(groups.length)} {groups.length === 1 ? names.one : names.many}{" "}
            {groups.length === 1 ? "has" : "have"} at least {orders(view.minOrders)}, worst first:
            by the input that pulls fit down most, compared with every order. Click one to see all
            eight inputs.
          </p>
          {groups.length === LIMIT && (
            <p className="note">Showing the worst {LIMIT}; raise the minimum to narrow it.</p>
          )}
          <div className="table-scroll">
            <table className="ranking">
              <thead>
                <tr>
                  <th scope="col">{capital(names.one)}</th>
                  <th scope="col">Orders</th>
                  <th scope="col">Fit</th>
                  <th scope="col">Worse here than over every order</th>
                </tr>
              </thead>
              <tbody>
                {groups.map((group) => (
                  <GroupRows
                    key={group.id}
                    group={group}
                    overallFit={fit}
                    open={open === group.id}
                    toggle={() => toggle(group.id)}
                  />
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </>
  );
}

function GroupRows({
  group,
  overallFit,
  open,
  toggle,
}: {
  group: DriverGroup;
  overallFit: number;
  open: boolean;
  toggle: () => void;
}) {
  const named = namedDrivers(group);
  return (
    <>
      <tr className={open ? "picked" : undefined}>
        <th scope="row">
          <button type="button" className="row-toggle" aria-expanded={open} onClick={toggle}>
            {group.id}
          </button>
        </th>
        <td>{whole(group.orders)}</td>
        <td>{decimal(group.fit)}</td>
        <td>
          {named.length === 0
            ? "Nothing"
            : named
                .map((input) => `${INPUT_NAMES[input]} ${signed(group.excess[input])}`)
                .join(", ")}
        </td>
      </tr>
      {open && (
        <tr className="detail">
          <td colSpan={4}>
            <p>
              {orders(group.orders)}, fit {decimal(group.fit)}; over every order,{" "}
              {decimal(overallFit)}.
            </p>
            <Bars
              parts={group.excess}
              label={`${group.id}: each input's share of fit, minus its share over every order`}
              floor={SHARE_FLOOR}
              format={(value) => signed(value)}
              better={(value) => value > 0}
              legend={
                <>
                  <span className="key better" /> better here than over every order{" "}
                  <span className="key worse" /> worse. Bars to the left pull fit down more here.
                </>
              }
            />
          </td>
        </tr>
      )}
    </>
  );
}
