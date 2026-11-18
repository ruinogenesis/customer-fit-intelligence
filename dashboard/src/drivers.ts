// What pulls fit down at each restaurant or courier (GET /drivers, T315), as
// the Drivers screen (T405) asks for it and reads it.
import type { Input } from "./customers";
import { addDays } from "./matrix";

export const BY = ["restaurant", "courier"] as const;
export type By = (typeof BY)[number];

export const BY_NAMES: Record<By, { one: string; many: string }> = {
  restaurant: { one: "restaurant", many: "restaurants" },
  courier: { one: "courier", many: "couriers" },
};

export function capital(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** The API's default span: the last 28 Bangkok days, today included. */
export const WINDOW = 28;
/** The quick spans, each ending on the span's last day. */
export const SPANS = [7, WINDOW, 90] as const;
export const MIN_ORDERS = [1, 5, 10, 20] as const;
/** The API's default minimum: fewer orders are mostly noise. */
export const DEFAULT_MIN = 5;
/** The API's most groups per answer. */
export const LIMIT = 200;

export interface DriverGroup {
  id: string;
  orders: number;
  fit: number;
  /** Each input's average contribution to its orders' fit. */
  drag: Record<Input, number>;
  /** drag minus the overall drag: below 0, worse here than overall. */
  excess: Record<Input, number>;
  /** Inputs worse here than overall, worst first (at most 3). */
  drivers: Input[];
}

export interface Drivers {
  by: By;
  from: string;
  to: string;
  min_orders: number;
  overall: { orders: number; fit: number | null; drag: Record<Input, number> };
  /** Worst first: by the most negative excess, then by id. */
  groups: DriverGroup[];
}

/** What the screen shows, as its URL keeps it. */
export interface DriversView {
  by: By;
  from: string;
  to: string;
  minOrders: number;
}

const DAY = /^\d{4}-\d{2}-\d{2}$/;

/** The view in the URL's search, each part missing or unreadable at its default. */
export function readView(search: URLSearchParams, today: string): DriversView {
  const by = search.get("by");
  const to = search.get("to") ?? "";
  const last = DAY.test(to) ? to : today;
  const from = search.get("from") ?? "";
  const min = Number(search.get("min_orders"));
  return {
    by: by === "courier" ? "courier" : "restaurant",
    from: DAY.test(from) ? from : addDays(last, 1 - WINDOW),
    to: last,
    minOrders: Number.isInteger(min) && min >= 1 ? min : DEFAULT_MIN,
  };
}

export function driversPath({ by, from, to, minOrders }: DriversView): string {
  const query = new URLSearchParams({
    by,
    from,
    to,
    min_orders: String(minOrders),
    limit: String(LIMIT),
  });
  return `/drivers?${query}`;
}

/** The drivers worth naming: those whose excess shows at three decimals. */
export function namedDrivers(group: DriverGroup): Input[] {
  return group.drivers.filter((input) => Math.abs(group.excess[input]) >= 5e-4);
}

/** Days from from to to, both included. */
export function spanDays(from: string, to: string): number {
  return Math.round((Date.parse(to) - Date.parse(from)) / 86_400_000) + 1;
}
