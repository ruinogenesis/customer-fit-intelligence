// The automation rules as GET /rules answers them (T406).

export type RuleStatus = "off" | "dry_run" | "test" | "live";
export type RuleKind = "service" | "marketing";

export interface Rule {
  rule_id: string;
  name: string;
  kind: RuleKind;
  status: RuleStatus;
  holdout_share: number;
  condition: Record<string, unknown>;
  action: Record<string, unknown>;
  caps: Record<string, unknown>;
  updated_at: string;
}

/** A rule's way from off to live (design doc), as the screen names it. */
export const STATUSES: readonly RuleStatus[] = ["off", "dry_run", "test", "live"];

export const STATUS_NAMES: Record<RuleStatus, string> = {
  off: "Off",
  dry_run: "Dry run",
  test: "Test",
  live: "Live",
};

export const STATUS_MEANING: Record<RuleStatus, string> = {
  off: "does nothing; this is the rule's kill switch",
  dry_run: "logs what it would do, and sends nothing",
  test: "acts, and holds back its holdout share to measure the effect",
  live: "acts, and keeps a holdout so its uplift stays measurable",
};

export const KIND_NAMES: Record<RuleKind, string> = {
  service: "Service",
  marketing: "Marketing",
};

/** A share as a whole percentage, e.g. 0.5 as 50%. */
export function percent(share: number): string {
  return `${Math.round(share * 100)}%`;
}
