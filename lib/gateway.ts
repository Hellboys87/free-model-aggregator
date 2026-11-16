// Dialling the model gateway.
//
// The gateway takes Chat Completions bodies at $OPENAI_BASE_URL and
// forwards them to the free models on the entrant's own key. It is
// always dialled non-streaming: a complete answer can be translated and
// replayed to the client in either wire, and a refusal can fail over to
// the next model before the client has seen a single byte.

type JsonRecord = Record<string, unknown>;

export interface GatewayAnswer {
  ok: true;
  status: number;
  body: JsonRecord;
}

export interface GatewayRefusal {
  ok: false;
  /** HTTP status of the refusal, or 503 when no answer came back. */
  status: number;
  /** The refusal's error code as a string, "" when it carries none. */
  code: string;
  message: string;
}

export type GatewayResult = GatewayAnswer | GatewayRefusal;

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** The gateway base may or may not end in /v1; the route hangs off it. */
export function chatCompletionsUrl(baseUrl: string): string {
  const trimmed = baseUrl.replace(/\/+$/, "");
  return trimmed.endsWith("/v1") ? `${trimmed}/chat/completions` : `${trimmed}/v1/chat/completions`;
}

function refusalCode(body: unknown): string {
  if (!isRecord(body)) return "";
  const error = isRecord(body.error) ? body.error : null;
  const raw = error !== null ? (error.code ?? error.type) : (body.code ?? body.type);
  if (typeof raw === "string") return raw.slice(0, 64);
  if (typeof raw === "number") return String(raw).slice(0, 64);
  return "";
}

function refusalMessage(body: unknown, status: number): string {
  if (isRecord(body) && isRecord(body.error) && typeof body.error.message === "string") {
    return body.error.message;
  }
  if (isRecord(body) && typeof body.message === "string") return body.message;
  return `the gateway answered HTTP ${status}`;
}

/**
 * One dial. A 2xx with a JSON object body is an answer; anything else —
 * a refusal status, an unparseable body, a network error, a timeout —
 * is a refusal the caller may fail over from.
 */
export async function dialChatCompletion(
  url: string,
  apiKey: string,
  body: JsonRecord,
  timeoutMs: number,
): Promise<GatewayResult> {
  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: {
        authorization: `Bearer ${apiKey}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch (err) {
    return {
      ok: false,
      status: 503,
      code: "upstream_unreachable",
      message: err instanceof Error ? err.message : "the gateway could not be reached",
    };
  }
  const text = await response.text();
  let parsed: unknown = null;
  try {
    parsed = JSON.parse(text);
  } catch {
    parsed = null;
  }
  if (response.ok && isRecord(parsed)) {
    return { ok: true, status: response.status, body: parsed };
  }
  if (response.ok) {
    return {
      ok: false,
      status: response.status,
      code: "invalid_upstream_response",
      message: "the gateway answered without a JSON object body",
    };
  }
  return {
    ok: false,
    status: response.status,
    code: refusalCode(parsed),
    message: refusalMessage(parsed, response.status),
  };
}
