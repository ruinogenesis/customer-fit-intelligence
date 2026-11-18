// T407: the API, called with the signed-in person's ID token.
import { expect, test } from "vitest";

import { ApiError, httpApi } from "../src/api";

function fakeFetch(answer: Response) {
  const sent: { url: string; init: RequestInit | undefined }[] = [];
  const send = async (url: RequestInfo | URL, init?: RequestInit) => {
    sent.push({ url: String(url), init });
    return answer;
  };
  return { sent, send: send as typeof fetch };
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

test("a GET goes to /api/v1 with the person's ID token, and gives the JSON", async () => {
  const { sent, send } = fakeFetch(json({ models: [] }));
  const api = httpApi(async () => "a-token", send);
  expect(await api.get("/models")).toEqual({ models: [] });
  expect(sent).toEqual([
    { url: "/api/v1/models", init: { method: "GET", headers: { Authorization: "Bearer a-token" } } },
  ]);
});

test("a POST is a POST", async () => {
  const { sent, send } = fakeFetch(json({ version: "t310-check" }));
  await httpApi(async () => "a-token", send).post("/models/t310-check/promote");
  expect(sent[0]?.init?.method).toBe("POST");
});

test("text() gives the answer as text, such as a model card", async () => {
  const { send } = fakeFetch(new Response("# Model card\n"));
  expect(await httpApi(async () => "a-token", send).text("/models/t112-sim/card")).toBe(
    "# Model card\n",
  );
});

test("a refusal carries the API's own message and status", async () => {
  const { send } = fakeFetch(json({ code: "conflict", message: "Not promoted: no file" }, 409));
  const failure = await httpApi(async () => "a-token", send)
    .post("/models/x/promote")
    .catch((error: unknown) => error);
  expect(failure).toBeInstanceOf(ApiError);
  expect(failure).toMatchObject({ status: 409, message: "Not promoted: no file" });
});

test("a refusal that is not JSON says the status", async () => {
  const { send } = fakeFetch(new Response("<html>", { status: 502, statusText: "Bad Gateway" }));
  await expect(httpApi(async () => "a-token", send).get("/models")).rejects.toMatchObject({
    status: 502,
    message: "The API answered 502 Bad Gateway",
  });
});
