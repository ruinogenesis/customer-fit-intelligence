// Stand-ins for Firebase Auth and the API, so the tests run the app alone.
import { act } from "@testing-library/react";

import { ApiError, type Api } from "../src/api";
import { SignInError, signInMessage, type AuthService, type Session } from "../src/auth";
import type { ModelVersion } from "../src/models";
import type { Role } from "../src/roles";

/** Firebase Auth's stand-in: accounts by email, and a session to report. */
export class FakeAuth implements AuthService {
  signedIn: Session | null = null;
  private listener: ((session: Session | null) => void) | null = null;
  private accounts: Record<string, { password: string; role: Role | null }>;

  constructor(accounts: Record<string, { password: string; role: Role | null }> = {}) {
    this.accounts = accounts;
  }

  watch(callback: (session: Session | null) => void): () => void {
    this.listener = callback;
    return () => {
      this.listener = null;
    };
  }

  /** What Firebase says when the page loads. */
  answer(session: Session | null): void {
    act(() => this.listener?.(session));
  }

  async signIn(email: string, password: string): Promise<void> {
    const account = this.accounts[email];
    if (account === undefined || account.password !== password) {
      throw new SignInError(signInMessage("auth/invalid-credential"));
    }
    this.signedIn = { email, role: account.role };
    this.listener?.(this.signedIn);
  }

  async signOut(): Promise<void> {
    this.signedIn = null;
    this.listener?.(null);
  }

  async token(): Promise<string> {
    return "a-token";
  }
}

/** Three versions, newest first: a candidate, the live one, a retired one. */
export function registry(): ModelVersion[] {
  const model = (version: string, status: ModelVersion["status"], churn: number) => ({
    version,
    status,
    path: `gs://cfi-test-models/models/${version}/model.json`,
    sha256: "0".repeat(64),
    metrics: { churn_auc: churn, reorder_auc: 0.711 },
    created_at: "2026-11-14T00:00:00Z",
    promoted_at: status === "candidate" ? null : "2026-11-18T11:00:00Z",
  });
  return [
    model("t310-check", "candidate", 0.8),
    model("t112-sim", "live", 0.789),
    model("t100-old", "retired", 0.75),
  ];
}

/** The API's stand-in: the registry in memory, every call recorded. */
export class FakeApi implements Api {
  calls: string[] = [];
  /** The next call fails with this, once. */
  failNext: ApiError | null = null;
  models: ModelVersion[];
  cards: Record<string, string>;
  /** Other GET answers by path; an ApiError there is thrown, as a refusal. */
  answers: Record<string, unknown> = {};

  constructor(models: ModelVersion[] = registry(), cards: Record<string, string> = {}) {
    this.models = models;
    this.cards = cards;
  }

  /** How many times path was asked for. */
  asked(path: string): number {
    return this.calls.filter((call) => call === `GET ${path}`).length;
  }

  private record(method: string, path: string): void {
    this.calls.push(`${method} ${path}`);
    const failure = this.failNext;
    this.failNext = null;
    if (failure !== null) throw failure;
  }

  async get<T>(path: string): Promise<T> {
    this.record("GET", path);
    if (path === "/models") return { models: structuredClone(this.models) } as T;
    const answer = this.answers[path];
    if (answer instanceof ApiError) throw answer;
    if (answer === undefined) throw new ApiError(404, `No ${path} here.`);
    return structuredClone(answer) as T;
  }

  async post<T>(path: string): Promise<T> {
    this.record("POST", path);
    const version = /^\/models\/([^/]+)\/promote$/.exec(path)?.[1];
    const target = this.models.find((model) => model.version === version);
    if (target === undefined) throw new ApiError(404, `No model version ${version}.`);
    for (const model of this.models) {
      if (model.status === "live" && model !== target) model.status = "retired";
    }
    target.status = "live";
    target.promoted_at = "2026-11-18T13:00:00Z";
    return structuredClone(target) as T;
  }

  async text(path: string): Promise<string> {
    this.record("GET", path);
    const version = /^\/models\/([^/]+)\/card$/.exec(path)?.[1] ?? "";
    const card = this.cards[version];
    if (card === undefined) throw new ApiError(404, `${version} has no model card.`);
    return card;
  }
}
