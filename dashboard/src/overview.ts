// The Overview screen's data (T402): GET /overview now, GET /overview/trend
// for each day up to yesterday, GET /customers for the riskiest.
export const QUADRANTS = ["lovemark", "brand", "fad", "product"] as const;
export type Quadrant = (typeof QUADRANTS)[number];
export type QuadrantCounts = Record<Quadrant, number>;

export const QUADRANT_NAMES: Record<Quadrant, string> = {
  lovemark: "Lovemark",
  brand: "Brand",
  fad: "Fad",
  product: "Product",
};

/** What each quadrant means: where Love and Respect sit (D5). */
export const QUADRANT_MEANING: Record<Quadrant, string> = {
  lovemark: "Loved and respected",
  brand: "Respected, not loved",
  fad: "Loved, not respected",
  product: "Neither loved nor respected",
};

export interface OverviewNow {
  customers: number;
  unscored: number;
  quadrants: QuadrantCounts;
  average_fit: number | null;
  average_churn_risk: number | null;
  at_risk: { threshold: number; customers: number };
}

export interface TrendDay {
  day: string;
  customers: number;
  average_fit: number;
  average_churn_risk: number;
  quadrants: QuadrantCounts;
}

export interface Trend {
  from: string;
  to: string;
  days: TrendDay[];
}

/** A customer as GET /customers lists them (the fields the screen uses). */
export interface ListedCustomer {
  customer_id: string;
  quadrant: Quadrant | null;
  churn_risk: number | null;
  scores: { fit: number } | null;
}

/** 0.0412 -> "4.1%"; null -> "—". */
export function percent(value: number | null, digits = 1): string {
  return value === null ? "—" : `${(value * 100).toFixed(digits)}%`;
}

/** 0.74213 -> "0.742"; null -> "—". */
export function decimal(value: number | null, digits = 3): string {
  return value === null ? "—" : value.toFixed(digits);
}

/** 12345 -> "12,345". */
export function whole(value: number): string {
  return value.toLocaleString("en-US");
}

/** A part of a whole as a share: 3 of 12 -> "25%"; nothing of nothing -> "—". */
export function share(part: number, whole: number): string {
  return whole === 0 ? "—" : `${Math.round((part / whole) * 100)}%`;
}

/** A moment as a Bangkok clock time, e.g. 20:51:03. */
export function clock(at: Date): string {
  return new Intl.DateTimeFormat("en-GB", {
    timeZone: "Asia/Bangkok",
    timeStyle: "medium",
  }).format(at);
}
