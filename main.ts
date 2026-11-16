// free-model-aggregator-02: a free-model aggregator.
//
// One server speaking both client wires (OpenAI Chat Completions and
// Anthropic Messages, JSON and streaming). For every prompt it picks the
// highest-rated free model that meets the prompt's requirements, dials
// it through the gateway on the entrant's key, and translates the answer
// back into the client's wire with the dialled model named. A refusal
// fails over to the next-best compliant model, and the refusals it
// recovered from are reported in the x-makefun-upstream-failures header.
//
//   node main.ts          run the trial server
//   node main.ts --list   print the catalog document and exit
//
// Every choice derives from the brief, the request, and the gateway's
// answers — no clock, no randomness — so a replayed trial repeats the
// same dials byte for byte.

import { createServer } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import { readFile } from "node:fs/promises";
import {
  buildCatalog,
  parseBrief,
  rankCandidates,
  type Brief,
  type FreeModel,
  type Requirements,
} from "./lib/brief.ts";
import {
  anthropicAnswerFromOpenAI,
  anthropicErrorBody,
  anthropicModelList,
  anthropicRequestToOpenAI,
  anthropicStreamFrames,
  openAIAnswerForClient,
  openAIErrorBody,
  openAIModelList,
  openAIRequestForGateway,
  openAIStreamFrames,
  requirementsOfBody,
} from "./lib/wire.ts";
import { chatCompletionsUrl, dialChatCompletion } from "./lib/gateway.ts";

type JsonRecord = Record<string, unknown>;
type WireApi = "openai" | "anthropic";

const MAX_INBOUND_BYTES = 8 * 1024 * 1024;
const MAX_ATTEMPTS_PER_PROMPT = 4;
const MAX_TOTAL_ATTEMPTS = 24;
const ATTEMPT_TIMEOUT_MS = 20_000;
const PROMPT_BUDGET_MS = 26_000;
const FAILURES_HEADER = "x-makefun-upstream-failures";

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requiredEnv(name: string): string {
  const value = process.env[name];
  if (value === undefined || value === "") throw new Error(`${name} is not set`);
  return value;
}

async function readBrief(): Promise<Brief> {
  const path = process.env.MAKEFUN_PROMPT_FILE;
  if (path === undefined || path === "") return { models: [], prompts: [] };
  try {
    return parseBrief(await readFile(path, "utf8"));
  } catch {
    return { models: [], prompts: [] };
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------
// HTTP plumbing.
// ---------------------------------------------------------------------

function sendJson(
  res: ServerResponse,
  status: number,
  value: unknown,
  headers: Record<string, string> = {},
): void {
  const body = Buffer.from(`${JSON.stringify(value)}\n`);
  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": String(body.length),
    ...headers,
  });
  res.end(body);
}

function sendStream(res: ServerResponse, frames: string[], headers: Record<string, string> = {}): void {
  res.writeHead(200, { "content-type": "text/event-stream", ...headers });
  res.end(frames.join(""));
}

async function readBody(req: IncomingMessage): Promise<Buffer> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk as Uint8Array);
    size += buffer.length;
    if (size > MAX_INBOUND_BYTES) throw new Error("the request body is too large");
    chunks.push(buffer);
  }
  return Buffer.concat(chunks);
}

// ---------------------------------------------------------------------
// The completion path: pick, dial, fail over, translate, answer.
// ---------------------------------------------------------------------

interface UpstreamFailure {
  status: number;
  code: string;
  model: string;
}

interface TrialState {
  brief: Brief;
  gatewayUrl: string;
  apiKey: string;
  /** Manifest rows are consumed in the order completion requests land. */
  nextPrompt: number;
  /** Models that refused earlier in this trial; tried last, not first. */
  refusedModels: Set<string>;
  totalAttempts: number;
}

function failuresHeader(failures: UpstreamFailure[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { [FAILURES_HEADER]: JSON.stringify(failures.slice(0, 8)) };
}

async function handleCompletion(state: TrialState, api: WireApi, req: IncomingMessage, res: ServerResponse): Promise<void> {
  let body: JsonRecord;
  try {
    const parsed: unknown = JSON.parse((await readBody(req)).toString("utf8"));
    if (!isRecord(parsed)) throw new Error("the request body is not one JSON object");
    body = parsed;
  } catch (err) {
    const message = err instanceof Error ? err.message : "the request body is not valid JSON";
    if (api === "openai") {
      sendJson(res, 400, { error: { message, type: "invalid_request_error", code: "invalid_request" } });
    } else {
      sendJson(res, 400, anthropicErrorBody(message, false));
    }
    return;
  }

  // Requirements: the brief manifest's row for this prompt, unioned with
  // whatever the body itself declares (JSON mode, tools, an image part).
  const spec = state.brief.prompts[state.nextPrompt];
  state.nextPrompt += 1;
  const requires: Requirements = { ...(spec !== undefined && spec.api === api ? spec.requires : {}) };
  const fromBody = requirementsOfBody(api, body);
  if (fromBody.tools === true) requires.tools = true;
  if (fromBody.vision === true) requires.vision = true;
  if (fromBody.json === true) requires.json = true;

  const gatewayBody =
    api === "openai" ? openAIRequestForGateway(body) : anthropicRequestToOpenAI(body);
  const wantStream = body.stream === true;

  const ranked = rankCandidates(state.brief.models, requires);
  const ordered: FreeModel[] = [
    ...ranked.filter((model) => !state.refusedModels.has(model.id)),
    ...ranked.filter((model) => state.refusedModels.has(model.id)),
  ];

  const failures: UpstreamFailure[] = [];
  const startedAt = Date.now();
  let attempts = 0;
  for (const candidate of ordered) {
    if (attempts >= MAX_ATTEMPTS_PER_PROMPT) break;
    if (state.totalAttempts >= MAX_TOTAL_ATTEMPTS) break;
    const elapsed = Date.now() - startedAt;
    if (attempts > 0 && elapsed > PROMPT_BUDGET_MS) break;
    const timeout = Math.max(1_000, Math.min(ATTEMPT_TIMEOUT_MS, PROMPT_BUDGET_MS - elapsed));
    attempts += 1;
    state.totalAttempts += 1;
    gatewayBody.model = candidate.id;
    const result = await dialChatCompletion(state.gatewayUrl, state.apiKey, gatewayBody, timeout);
    if (result.ok) {
      const headers = failuresHeader(failures);
      if (api === "openai") {
        if (wantStream) sendStream(res, openAIStreamFrames(result.body, candidate.id), headers);
        else sendJson(res, 200, openAIAnswerForClient(result.body, candidate.id), headers);
      } else {
        if (wantStream) sendStream(res, anthropicStreamFrames(result.body, candidate.id), headers);
        else sendJson(res, 200, anthropicAnswerFromOpenAI(result.body, candidate.id), headers);
      }
      return;
    }
    failures.push({ status: result.status, code: result.code, model: candidate.id });
    state.refusedModels.add(candidate.id);
    log(`model ${candidate.id} refused (${result.status} ${result.code}): ${result.message}`);
  }

  const message =
    failures.length === 0
      ? "no free model meets this prompt's requirements"
      : `every candidate model refused; last refusal: HTTP ${failures[failures.length - 1].status}`;
  if (api === "openai") {
    sendJson(res, 502, openAIErrorBody(message, failures.length > 0 ? failures[failures.length - 1].code : null), failuresHeader(failures));
  } else {
    sendJson(res, 502, anthropicErrorBody(message, true), failuresHeader(failures));
  }
}

function handleModelList(state: TrialState, req: IncomingMessage, res: ServerResponse): void {
  // The auth header names the wire: an Anthropic key header gets the
  // Anthropic list shape, a bearer token the OpenAI one.
  if (typeof req.headers["x-api-key"] === "string") {
    sendJson(res, 200, anthropicModelList(state.brief.models));
  } else {
    sendJson(res, 200, openAIModelList(state.brief.models));
  }
}

// ---------------------------------------------------------------------
// Trial lifecycle: listen, register the catalog, serve, poll for done.
// ---------------------------------------------------------------------

async function registerCatalog(envUrl: string, port: number, catalog: unknown): Promise<void> {
  const body = JSON.stringify({ protocol_version: 1, port, catalog });
  let lastError = "no attempt made";
  for (let attempt = 0; attempt < 40; attempt += 1) {
    try {
      const response = await fetch(`${envUrl}/register`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body,
        signal: AbortSignal.timeout(2_000),
      });
      if (response.ok) return;
      lastError = `the environment answered HTTP ${response.status}`;
    } catch (err) {
      lastError = err instanceof Error ? err.message : "the environment could not be reached";
    }
    await sleep(250);
  }
  throw new Error(`could not register with the environment: ${lastError}`);
}

async function pollWorkUntilDone(envUrl: string): Promise<void> {
  for (;;) {
    try {
      const response = await fetch(`${envUrl}/work`, { signal: AbortSignal.timeout(2_000) });
      if (response.ok) {
        const work: unknown = await response.json();
        if (isRecord(work) && work.done === true) return;
      }
    } catch {
      // The environment is the trial's clock; keep polling until it
      // answers done or the trial wall ends the process.
    }
    await sleep(100);
  }
}

async function main(): Promise<void> {
  const brief = await readBrief();
  const catalog = buildCatalog(brief.models);

  if (process.argv.includes("--list")) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return;
  }

  const envUrl = requiredEnv("MAKEFUN_ENV_URL").replace(/\/+$/, "");
  const state: TrialState = {
    brief,
    gatewayUrl: chatCompletionsUrl(requiredEnv("OPENAI_BASE_URL")),
    apiKey: process.env.OPENAI_API_KEY ?? "",
    nextPrompt: 0,
    refusedModels: new Set<string>(),
    totalAttempts: 0,
  };

  const server = createServer((req, res) => {
    const path = (req.url ?? "").split("?")[0];
    const route = `${req.method} ${path}`;
    const run = async (): Promise<void> => {
      if (route === "GET /v1/models") handleModelList(state, req, res);
      else if (route === "POST /v1/chat/completions") await handleCompletion(state, "openai", req, res);
      else if (route === "POST /v1/messages") await handleCompletion(state, "anthropic", req, res);
      else sendJson(res, 404, { error: { message: `no route ${route}`, type: "invalid_request_error", code: "not_found" } });
    };
    run().catch((err: unknown) => {
      log(`request failed: ${err instanceof Error ? err.message : String(err)}`);
      if (!res.headersSent) {
        sendJson(res, 500, { error: { message: "internal error", type: "api_error", code: "internal" } });
      } else {
        res.end();
      }
    });
  });

  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "0.0.0.0", () => resolve());
  });
  const address = server.address();
  const port = isRecord(address) && typeof address.port === "number" ? address.port : 0;
  if (port === 0 || port === 8080) throw new Error("could not take a free port of my own");

  await registerCatalog(envUrl, port, catalog);
  log(`registered on port ${port} with ${brief.models.length} free models`);
  await pollWorkUntilDone(envUrl);
  server.close();
}

process.on("SIGTERM", () => process.exit(0));
process.on("SIGINT", () => process.exit(0));

main().then(
  () => process.exit(0),
  (err: unknown) => {
    process.stderr.write(`aggregator: ${err instanceof Error ? err.message : String(err)}\n`);
    process.exit(1);
  },
);
