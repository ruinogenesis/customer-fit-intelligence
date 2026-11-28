// The Automations screen (T406): every rule with its kind, status and holdout
// share. Read-only until the rules engine (T501) can act on a rule.
import { useEffect, useState } from "react";

import { bangkok } from "./models";
import { atLeast, whoMay } from "./roles";
import { KIND_NAMES, percent, STATUS_MEANING, STATUS_NAMES, STATUSES, type Rule } from "./rules";
import { useStaff } from "./staff";

type Loaded =
  | { state: "loading" }
  | { state: "failed"; message: string }
  | { state: "ready"; rules: Rule[] };

export function AutomationsPage() {
  const { role, api } = useStaff();
  const admin = atLeast(role, "admin");
  const [loaded, setLoaded] = useState<Loaded>({ state: "loading" });

  useEffect(() => {
    let current = true;
    api.get<{ rules: Rule[] }>("/rules").then(
      ({ rules }) => {
        if (current) setLoaded({ state: "ready", rules });
      },
      (failure: unknown) => {
        const message = failure instanceof Error ? failure.message : String(failure);
        if (current) setLoaded({ state: "failed", message });
      },
    );
    return () => {
      current = false;
    };
  }, [api]);

  return (
    <section className="automations">
      <h1>Automations</h1>
      <p className="lede">
        Rules, their status, and the share of customers each holds back to measure its effect.
      </p>
      {loaded.state === "loading" && <p>Loading the rules…</p>}
      {loaded.state === "failed" && <p role="alert">{loaded.message}</p>}
      {loaded.state === "ready" && loaded.rules.length === 0 && (
        <p>
          No rules yet. An admin adds the six starter rules with{" "}
          <code>python -m cfi_api.rules seed</code>.
        </p>
      )}
      {loaded.state === "ready" && loaded.rules.length > 0 && <Rules rules={loaded.rules} />}
      <p className={admin ? "action" : "action read-only"}>
        {admin
          ? "Turning rules on or off, and promoting one from test to live, comes with the rules engine (T501)."
          : `Only ${whoMay("admin")} will be able to turn rules on or off, or promote one from test to live (T501).`}
      </p>
    </section>
  );
}

function Rules({ rules }: { rules: Rule[] }) {
  return (
    <>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Rule</th>
              <th>Kind</th>
              <th>Status</th>
              <th>Holdout</th>
              <th>Last changed</th>
            </tr>
          </thead>
          <tbody>
            {rules.map((rule) => (
              <tr key={rule.rule_id}>
                <td>{rule.name}</td>
                <td>{KIND_NAMES[rule.kind]}</td>
                <td>
                  <span className={`status ${rule.status}`}>{STATUS_NAMES[rule.status]}</span>
                </td>
                <td>{percent(rule.holdout_share)}</td>
                <td>{bangkok(rule.updated_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ul className="note statuses">
        {STATUSES.map((status) => (
          <li key={status}>
            <b>{STATUS_NAMES[status]}</b>: {STATUS_MEANING[status]}.
          </li>
        ))}
        <li>Service rules skip the marketing consent check and the frequency cap.</li>
      </ul>
    </>
  );
}
