// The model registry as GET /models answers it (T310).

export type Status = "candidate" | "live" | "retired";

export interface ModelVersion {
  version: string;
  status: Status;
  path: string;
  sha256: string;
  metrics: Record<string, number>;
  created_at: string;
  promoted_at: string | null;
}

/** What promoting this version does, as its button says it. */
export function promoteLabel(model: ModelVersion): string | null {
  if (model.status === "live") return null;
  return model.status === "retired" ? "Roll back to this" : "Promote";
}

/** Every metric name across the versions, sorted: the table's columns. */
export function metricNames(models: readonly ModelVersion[]): string[] {
  return [...new Set(models.flatMap((model) => Object.keys(model.metrics)))].sort();
}

/** A UTC timestamp as a Bangkok date and time, e.g. 2026-11-18 19:52. */
export function bangkok(at: string | null): string {
  if (at === null) return "—";
  return new Intl.DateTimeFormat("sv-SE", {
    timeZone: "Asia/Bangkok",
    dateStyle: "short",
    timeStyle: "short",
  }).format(new Date(at));
}
