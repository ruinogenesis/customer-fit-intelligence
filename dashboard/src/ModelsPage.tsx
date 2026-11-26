// The Models screen (T407): every version with its status, metrics and model
// card; admins promote a candidate or roll back to an earlier version.
import { Fragment, useCallback, useEffect, useState } from "react";

import { ApiError } from "./api";
import { toSafeHtml } from "./markdown";
import { bangkok, metricNames, promoteLabel, type ModelVersion } from "./models";
import { atLeast } from "./roles";
import { useStaff } from "./staff";

type Registry =
  | { state: "loading" }
  | { state: "failed"; message: string }
  | { state: "ready"; models: ModelVersion[] };

type Card = { html: string } | { message: string };

function messageOf(failure: unknown): string {
  return failure instanceof Error ? failure.message : String(failure);
}

export function ModelsPage() {
  const { role, api } = useStaff();
  const admin = atLeast(role, "admin");
  const [registry, setRegistry] = useState<Registry>({ state: "loading" });
  const [confirming, setConfirming] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [cards, setCards] = useState<Record<string, Card>>({});

  const reload = useCallback(async () => {
    try {
      const { models } = await api.get<{ models: ModelVersion[] }>("/models");
      setRegistry({ state: "ready", models });
    } catch (failure) {
      setRegistry({ state: "failed", message: messageOf(failure) });
    }
  }, [api]);

  useEffect(() => {
    void reload();
  }, [reload]);

  async function promote(version: string) {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      await api.post(`/models/${encodeURIComponent(version)}/promote`);
      setConfirming(null);
      setNotice(`${version} is live. The next event on every instance is scored by it.`);
      await reload();
    } catch (failure) {
      setError(messageOf(failure));
    } finally {
      setBusy(false);
    }
  }

  async function toggleCard(version: string) {
    if (open === version) {
      setOpen(null);
      return;
    }
    setOpen(version);
    if (cards[version] !== undefined) return;
    let card: Card;
    try {
      card = { html: toSafeHtml(await api.text(`/models/${encodeURIComponent(version)}/card`)) };
    } catch (failure) {
      const missing = failure instanceof ApiError && failure.status === 404;
      card = { message: missing ? "This version has no model card." : messageOf(failure) };
    }
    setCards((known) => ({ ...known, [version]: card }));
  }

  return (
    <section className="models">
      <h1>Models</h1>
      <p className="lede">
        Versions, metrics and model cards.{" "}
        {admin
          ? "Promote a candidate, or roll back to an earlier version."
          : "Only admins can promote or roll back."}
      </p>
      {notice !== null && (
        <p role="status" className="notice">
          {notice}
        </p>
      )}
      {error !== null && <p role="alert">{error}</p>}
      {registry.state === "loading" && <p>Loading the registry…</p>}
      {registry.state === "failed" && <p role="alert">{registry.message}</p>}
      {registry.state === "ready" && registry.models.length === 0 && (
        <p>No model versions are registered yet.</p>
      )}
      {registry.state === "ready" && registry.models.length > 0 && (
        <Versions
          models={registry.models}
          admin={admin}
          confirming={confirming}
          busy={busy}
          open={open}
          cards={cards}
          onPromote={(version) => {
            setError(null);
            setConfirming(version);
          }}
          onCancel={() => setConfirming(null)}
          onConfirm={(version) => void promote(version)}
          onCard={(version) => void toggleCard(version)}
        />
      )}
    </section>
  );
}

function Versions(props: {
  models: ModelVersion[];
  admin: boolean;
  confirming: string | null;
  busy: boolean;
  open: string | null;
  cards: Record<string, Card>;
  onPromote: (version: string) => void;
  onCancel: () => void;
  onConfirm: (version: string) => void;
  onCard: (version: string) => void;
}) {
  const { models, admin, confirming, busy, open, cards } = props;
  const cancel = (version: string) => {
    props.onCancel();
    document.getElementById(`promote-${version}`)?.focus();
  };
  const metrics = metricNames(models);
  const live = models.find((model) => model.status === "live")?.version ?? null;
  const columns = 5 + metrics.length;
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Version</th>
            <th>Status</th>
            {metrics.map((name) => (
              <th key={name}>{name}</th>
            ))}
            <th>Trained</th>
            <th>Promoted</th>
            <th>
              <span className="visually-hidden">Actions</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {models.map((model) => {
            const label = promoteLabel(model);
            const card = cards[model.version];
            return (
              <Fragment key={model.version}>
                <tr className={model.status}>
                  <td>
                    <code>{model.version}</code>
                  </td>
                  <td>
                    <span className={`status ${model.status}`}>{model.status}</span>
                  </td>
                  {metrics.map((name) => (
                    <td key={name}>{model.metrics[name]?.toFixed(3) ?? "—"}</td>
                  ))}
                  <td>{bangkok(model.created_at)}</td>
                  <td>{bangkok(model.promoted_at)}</td>
                  <td className="actions">
                    <button
                      type="button"
                      aria-label={`Model card for ${model.version}`}
                      aria-expanded={open === model.version}
                      onClick={() => props.onCard(model.version)}
                    >
                      {open === model.version ? "Hide card" : "Model card"}
                    </button>
                    {admin && label !== null && (
                      <button
                        type="button"
                        id={`promote-${model.version}`}
                        aria-label={`${label} ${model.version}`}
                        aria-expanded={confirming === model.version}
                        disabled={busy}
                        onClick={() => props.onPromote(model.version)}
                      >
                        {label}
                      </button>
                    )}
                  </td>
                </tr>
                {confirming === model.version && (
                  <tr className="confirm">
                    {/* Focus goes to the confirm button; Cancel or Escape brings it
                        back to the button that asked (T409). */}
                    <td
                      colSpan={columns}
                      onKeyDown={(event) => {
                        if (event.key === "Escape" && !busy) cancel(model.version);
                      }}
                    >
                      <p id={`confirm-${model.version}`}>
                        Make <code>{model.version}</code> live? Every event from then on, on every
                        instance, is scored by it
                        {live === null ? "." : <>; <code>{live}</code> is retired.</>}
                      </p>
                      <button
                        type="button"
                        className="primary"
                        autoFocus
                        aria-describedby={`confirm-${model.version}`}
                        disabled={busy}
                        onClick={() => props.onConfirm(model.version)}
                      >
                        {busy ? "Promoting…" : `Make ${model.version} live`}
                      </button>
                      <button type="button" disabled={busy} onClick={() => cancel(model.version)}>
                        Cancel
                      </button>
                    </td>
                  </tr>
                )}
                {open === model.version && (
                  <tr className="card-row">
                    <td colSpan={columns}>
                      {card === undefined ? (
                        <p>Loading the model card…</p>
                      ) : "html" in card ? (
                        <div className="card" dangerouslySetInnerHTML={{ __html: card.html }} />
                      ) : (
                        <p>{card.message}</p>
                      )}
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
