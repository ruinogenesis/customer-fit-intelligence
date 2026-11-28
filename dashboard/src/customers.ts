// One customer as the API gives them (T311, T316): their state, every scoring
// with its exact split (T115), and their orders; and the sums the profile
// screen (T404) draws from them.
import type { Quadrant } from "./overview";

export const INPUTS = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"] as const;
export type Input = (typeof INPUTS)[number];

/** The model's inputs as the design doc names them. */
export const INPUT_NAMES: Record<Input, string> = {
  x1: "Speed vs promise",
  x2: "App ease",
  x3: "Fee load",
  x4: "Order fidelity",
  x5: "Courier conduct",
  x6: "Food & selection",
  x7: "Personalization & delight",
  x8: "Recovery handling",
};

export const SCORES = ["fit", "love", "respect", "trust", "convenience", "churn_risk"] as const;
export type Score = (typeof SCORES)[number];

export const SCORE_NAMES: Record<Score, string> = {
  fit: "Fit",
  love: "Love",
  respect: "Respect",
  trust: "Trust",
  convenience: "Convenience",
  churn_risk: "Churn risk",
};

export type Latents = Record<Exclude<Score, "churn_risk">, number>;

/** A customer's latest marketing consent (T610). */
export interface Consent {
  marketing: boolean;
  at: string;
  source: "sign_up" | "settings" | "support";
}

/** customers/{id}, as GET /customers/{id} answers it. */
export interface CustomerState {
  customer_id: string;
  latest_order_id: string | null;
  latest_placed_at: string | null;
  scores: Latents | null;
  churn_risk: number | null;
  quadrant: Quadrant | null;
  model_version: string;
  /** Null when never asked; missing on customers saved before T610. */
  consent?: Consent | null;
}

/** One scoring of one order (customers/{id}/scores/{event_id}). */
export interface Scoring {
  event_id: string;
  order_id: string;
  at: string;
  scores: Latents;
  churn_risk: number;
  quadrant: Quadrant;
  /** Each input's share of each score's distance from "as expected"; absent before T316. */
  contributions?: Record<Score, Record<Input, number>>;
  /** Fit's biggest movers, biggest first; absent before T316. */
  drivers?: Input[];
  model_version: string;
}

export type IssueType = "missing" | "wrong" | "cold" | "late";

/** orders/{id}: its record and the gaps it has measured so far. */
export interface Order {
  order_id: string;
  customer_id: string;
  restaurant_id: string;
  placed_at: string;
  promised_eta: string;
  basket_value: number;
  fees: number;
  promo_used: boolean;
  courier_id: string | null;
  delivered_at: string | null;
  issues: { issue_type: IssueType; at: string }[];
  ratings: Partial<Record<"food" | "courier" | "accuracy" | "overall", number>>;
  gaps: Partial<Record<Input, number>>;
}

export interface ScoringPage {
  scores: Scoring[];
  next_cursor: string | null;
}

export interface OrderPage {
  orders: Order[];
  next_cursor: string | null;
}

/** The path of a customer, or of one of their lists, in the API and the app. */
export function customerPath(customerId: string, list = ""): string {
  return `/customers/${encodeURIComponent(customerId)}${list}`;
}

/** A score's value on a scoring: churn risk sits beside the latents. */
export function valueOf(scoring: Scoring, score: Score): number {
  return score === "churn_risk" ? scoring.churn_risk : scoring.scores[score];
}

/**
 * Why a score sits where it does: "as expected" (every gap 0) plus each
 * input's share, which add up to the score exactly. Null when the scoring
 * carries no split (scored before T316).
 */
export interface Split {
  expected: number;
  value: number;
  parts: Record<Input, number>;
}

export function splitOf(scoring: Scoring, score: Score): Split | null {
  const parts = scoring.contributions?.[score];
  if (parts === undefined) return null;
  const value = valueOf(scoring, score);
  const moved = INPUTS.reduce((sum, input) => sum + parts[input], 0);
  return { expected: value - moved, value, parts: { ...parts } };
}

/**
 * What moved a score from one scoring to the next: each input's share then
 * and now subtract (T115). The parts add up to the change only when both
 * scorings used the same model, whose "as expected" is then the same.
 */
export interface Change {
  from: number;
  to: number;
  parts: Record<Input, number>;
  sameModel: boolean;
}

export function changeOf(before: Scoring, after: Scoring, score: Score): Change | null {
  const then = splitOf(before, score);
  const now = splitOf(after, score);
  if (then === null || now === null) return null;
  const parts = Object.fromEntries(
    INPUTS.map((input) => [input, now.parts[input] - then.parts[input]]),
  ) as Record<Input, number>;
  return {
    from: then.value,
    to: now.value,
    parts,
    sameModel: before.model_version === after.model_version,
  };
}

/** Newest first, by time then event id, as the API sorts a customer's scorings. */
export function newestFirst(a: Scoring, b: Scoring): number {
  return a.at === b.at ? b.event_id.localeCompare(a.event_id) : b.at.localeCompare(a.at);
}

/** New scorings merged into the ones already held: one per event, newest first. */
export function mergeScorings(held: readonly Scoring[], fresh: readonly Scoring[]): Scoring[] {
  const byEvent = new Map(held.map((s) => [s.event_id, s]));
  for (const s of fresh) byEvent.set(s.event_id, s);
  return [...byEvent.values()].sort(newestFirst);
}

/** A moment in Bangkok, to the minute: 2026-03-01 19:51. */
export function when(iso: string): string {
  return new Intl.DateTimeFormat("sv-SE", {
    timeZone: "Asia/Bangkok",
    dateStyle: "short",
    timeStyle: "short",
  }).format(new Date(iso));
}

const CONSENT_SOURCES: Record<Consent["source"], string> = {
  sign_up: "at sign-up",
  settings: "in settings",
  support: "through support",
};

/** A customer's marketing consent in words, and whether marketing rules may
 * act for them (T610). Never asked counts as no. */
export function consentLine(consent: Consent | null | undefined): string {
  if (consent == null) return "Marketing consent: never asked, so no marketing rule acts for them.";
  const answer = consent.marketing ? "given" : consent.source === "sign_up" ? "declined" : "withdrawn";
  const line = `Marketing consent: ${answer} ${CONSENT_SOURCES[consent.source]}, ${when(consent.at)}.`;
  return consent.marketing ? line : `${line} No marketing rule acts for them.`;
}

/** A count and its noun, singular for 1: "1 order", "3 orders". */
export function count(n: number, noun: string): string {
  return `${n} ${noun}${n === 1 ? "" : "s"}`;
}

/** Satang as baht: 19118 -> "฿191.18". */
export function baht(satang: number): string {
  const amount = (satang / 100).toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
  return `฿${amount}`;
}

/** Delivery against the promise: "11 min late", "4 min early", "on time", "not delivered". */
export function delivery(order: Order): string {
  if (order.delivered_at === null) return "not delivered";
  const minutes = Math.round(
    (Date.parse(order.delivered_at) - Date.parse(order.promised_eta)) / 60_000,
  );
  if (minutes === 0) return "on time";
  return minutes > 0 ? `${minutes} min late` : `${-minutes} min early`;
}

/** A share or a change, signed: +0.042, −0.114 (a true minus), 0. */
export function signed(value: number, digits = 3): string {
  const text = Math.abs(value).toFixed(digits);
  if (Number(text) === 0) return (0).toFixed(digits);
  return `${value > 0 ? "+" : "−"}${text}`;
}
