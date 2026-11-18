// The API, called as the signed-in person: same origin (/api/** is Hosting's
// rewrite to Cloud Run; the dev server proxies it), with their ID token.

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export interface Api {
  get<T>(path: string): Promise<T>;
  post<T>(path: string): Promise<T>;
  /** A text answer, such as a model card's markdown. */
  text(path: string): Promise<string>;
}

export const BASE = "/api/v1";

export function httpApi(token: () => Promise<string>, send: typeof fetch = fetch): Api {
  async function call(method: string, path: string): Promise<Response> {
    const answer = await send(`${BASE}${path}`, {
      method,
      headers: { Authorization: `Bearer ${await token()}` },
    });
    if (!answer.ok) throw new ApiError(answer.status, await reason(answer));
    return answer;
  }
  return {
    get: async (path) => (await call("GET", path)).json(),
    post: async (path) => (await call("POST", path)).json(),
    text: async (path) => (await call("GET", path)).text(),
  };
}

/** The API's own message ({code, message}), or the status in words. */
async function reason(answer: Response): Promise<string> {
  try {
    const body: unknown = await answer.json();
    if (typeof body === "object" && body !== null && "message" in body) {
      return String(body.message);
    }
  } catch {
    // Not JSON: fall through.
  }
  return `The API answered ${answer.status} ${answer.statusText}`.trim();
}
