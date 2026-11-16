// The two client wires and the translations between them.
//
// Clients speak either the OpenAI Chat Completions wire or the Anthropic
// Messages wire. The gateway only takes Chat Completions bodies, so an
// Anthropic request is translated into one before dialling and the
// answer is translated back. Every answer is dialled non-streaming (see
// gateway.ts); when the client asked for a stream, the complete answer
// is replayed as a well-formed SSE stream in the client's own wire, with
// the event order each wire's checks demand.
//
// Deterministic throughout: ids and timestamps in synthesized output
// come from the recorded gateway answer, never from the clock.

import type { FreeModel, Requirements } from "./brief.ts";

type JsonRecord = Record<string, unknown>;

/** Fixed list timestamps: a constant keeps --list and trials replayable. */
const LIST_CREATED = 1767225600; // 2026-01-01T00:00:00Z
const LIST_CREATED_AT = "2026-01-01T00:00:00Z";

/** The gateway refuses a completion with no token cap, or one over this. */
export const MAX_TOKEN_CAP = 8192;
const DEFAULT_TOKEN_CAP = 1024;

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function clampTokenCap(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  return Math.min(MAX_TOKEN_CAP, Math.max(1, Math.floor(value)));
}

/** Concatenate the text of a string or a content-block array. */
export function textOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  let text = "";
  for (const part of content) {
    if (isRecord(part) && part.type === "text" && typeof part.text === "string") {
      text += part.text;
    }
  }
  return text;
}

/**
 * What a request body itself demands, read off the body: a JSON-mode
 * response format, a declared tool list, an image part. Merged with the
 * brief manifest's requirements by the caller as a belt-and-braces union.
 */
export function requirementsOfBody(api: "openai" | "anthropic", body: JsonRecord): Requirements {
  const requires: Requirements = {};
  if (Array.isArray(body.tools) && body.tools.length > 0) requires.tools = true;
  if (api === "openai" && isRecord(body.response_format)) {
    if (body.response_format.type === "json_object" || body.response_format.type === "json_schema") {
      requires.json = true;
    }
  }
  const imageType = api === "openai" ? "image_url" : "image";
  if (Array.isArray(body.messages)) {
    outer: for (const message of body.messages) {
      if (!isRecord(message) || !Array.isArray(message.content)) continue;
      for (const part of message.content) {
        if (isRecord(part) && part.type === imageType) {
          requires.vision = true;
          break outer;
        }
      }
    }
  }
  return requires;
}

// ---------------------------------------------------------------------
// Request translation: client body -> gateway Chat Completions body.
// The model field is left for the caller, which dials one candidate at
// a time. Stream controls are dropped: the gateway is always dialled
// non-streaming and the client stream is synthesized from the answer.
// ---------------------------------------------------------------------

export function openAIRequestForGateway(body: JsonRecord): JsonRecord {
  const out: JsonRecord = { ...body };
  delete out.stream;
  delete out.stream_options;
  const cap = clampTokenCap(out.max_tokens);
  const completionCap = clampTokenCap(out.max_completion_tokens);
  if (cap !== null) out.max_tokens = cap;
  else delete out.max_tokens;
  if (completionCap !== null) out.max_completion_tokens = completionCap;
  else delete out.max_completion_tokens;
  if (cap === null && completionCap === null) out.max_tokens = DEFAULT_TOKEN_CAP;
  return out;
}

function anthropicImagePart(part: JsonRecord): JsonRecord | null {
  const source = isRecord(part.source) ? part.source : null;
  if (source === null || source.type !== "base64") return null;
  if (typeof source.data !== "string" || typeof source.media_type !== "string") return null;
  return {
    type: "image_url",
    image_url: { url: `data:${source.media_type};base64,${source.data}` },
  };
}

function anthropicToolChoice(choice: unknown): unknown {
  if (typeof choice === "string") return choice;
  if (!isRecord(choice)) return undefined;
  if (choice.type === "auto") return "auto";
  if (choice.type === "any") return "required";
  if (choice.type === "none") return "none";
  if (choice.type === "tool" && typeof choice.name === "string") {
    return { type: "function", function: { name: choice.name } };
  }
  return undefined;
}

export function anthropicRequestToOpenAI(body: JsonRecord): JsonRecord {
  const messages: unknown[] = [];
  if (body.system !== undefined) {
    messages.push({ role: "system", content: textOf(body.system) });
  }
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of rawMessages) {
    if (!isRecord(raw) || typeof raw.role !== "string") continue;
    if (typeof raw.content === "string") {
      messages.push({ role: raw.role, content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    if (raw.role === "assistant") {
      let text = "";
      const toolCalls: JsonRecord[] = [];
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === "text" && typeof part.text === "string") text += part.text;
        else if (part.type === "tool_use" && typeof part.name === "string") {
          toolCalls.push({
            id: typeof part.id === "string" ? part.id : `call_${toolCalls.length}`,
            type: "function",
            function: { name: part.name, arguments: JSON.stringify(part.input ?? {}) },
          });
        }
      }
      const message: JsonRecord = { role: "assistant", content: text === "" && toolCalls.length > 0 ? null : text };
      if (toolCalls.length > 0) message.tool_calls = toolCalls;
      messages.push(message);
      continue;
    }
    // A user turn: tool results become tool messages, the rest stays a
    // user message with content parts.
    const parts: unknown[] = [];
    for (const part of raw.content) {
      if (!isRecord(part)) continue;
      if (part.type === "tool_result") {
        messages.push({
          role: "tool",
          tool_call_id: typeof part.tool_use_id === "string" ? part.tool_use_id : "",
          content: textOf(part.content),
        });
      } else if (part.type === "text" && typeof part.text === "string") {
        parts.push({ type: "text", text: part.text });
      } else if (part.type === "image") {
        const image = anthropicImagePart(part);
        if (image !== null) parts.push(image);
      }
    }
    if (parts.length > 0) messages.push({ role: "user", content: parts });
  }

  const out: JsonRecord = { messages };
  out.max_tokens = clampTokenCap(body.max_tokens) ?? DEFAULT_TOKEN_CAP;
  if (typeof body.temperature === "number") out.temperature = body.temperature;
  if (typeof body.top_p === "number") out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences)) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    out.tools = body.tools.filter(isRecord).map((tool) => {
      const fn: JsonRecord = { name: typeof tool.name === "string" ? tool.name : "" };
      if (typeof tool.description === "string") fn.description = tool.description;
      fn.parameters = isRecord(tool.input_schema) ? tool.input_schema : { type: "object", properties: {} };
      return { type: "function", function: fn };
    });
  }
  const toolChoice = anthropicToolChoice(body.tool_choice);
  if (toolChoice !== undefined) out.tool_choice = toolChoice;
  return out;
}

// ---------------------------------------------------------------------
// Answer translation: gateway Chat Completions answer -> client wire.
// The model field always names the id actually dialled.
// ---------------------------------------------------------------------

/** The OpenAI answer, relayed whole with the dialled model restored. */
export function openAIAnswerForClient(answer: JsonRecord, dialledId: string): JsonRecord {
  return { ...answer, model: dialledId };
}

function parseToolArguments(raw: unknown): JsonRecord {
  if (isRecord(raw)) return raw;
  if (typeof raw !== "string") return {};
  try {
    const parsed: unknown = JSON.parse(raw);
    return isRecord(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function anthropicStopReason(finishReason: unknown): string {
  switch (finishReason) {
    case "length":
      return "max_tokens";
    case "tool_calls":
    case "function_call":
      return "tool_use";
    case "content_filter":
      return "refusal";
    case "stop_sequence":
      return "stop_sequence";
    default:
      return "end_turn";
  }
}

/** Content blocks of the translated Anthropic answer, in answer order. */
function anthropicContentOf(answer: JsonRecord): JsonRecord[] {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const blocks: JsonRecord[] = [];
  const text = textOf(message.content);
  const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  if (text !== "" || toolCalls.length === 0) blocks.push({ type: "text", text });
  toolCalls.forEach((rawCall, index) => {
    if (!isRecord(rawCall) || !isRecord(rawCall.function)) return;
    blocks.push({
      type: "tool_use",
      id: typeof rawCall.id === "string" ? rawCall.id : `toolu_${index}`,
      name: typeof rawCall.function.name === "string" ? rawCall.function.name : "",
      input: parseToolArguments(rawCall.function.arguments),
    });
  });
  return blocks;
}

function usageOf(answer: JsonRecord): { prompt: number; completion: number } {
  const usage = isRecord(answer.usage) ? answer.usage : {};
  return {
    prompt: typeof usage.prompt_tokens === "number" ? usage.prompt_tokens : 0,
    completion: typeof usage.completion_tokens === "number" ? usage.completion_tokens : 0,
  };
}

export function anthropicAnswerFromOpenAI(answer: JsonRecord, dialledId: string): JsonRecord {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const usage = usageOf(answer);
  return {
    id: `msg_${typeof answer.id === "string" ? answer.id : "answer"}`,
    type: "message",
    role: "assistant",
    model: dialledId,
    content: anthropicContentOf(answer),
    stop_reason: anthropicStopReason(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: usage.prompt, output_tokens: usage.completion },
  };
}

// ---------------------------------------------------------------------
// Streaming synthesis: the complete answer replayed as SSE frames.
// ---------------------------------------------------------------------

function sseData(value: unknown): string {
  return `data: ${JSON.stringify(value)}\n\n`;
}

function sseEvent(name: string, value: unknown): string {
  return `event: ${name}\n${sseData(value)}`;
}

/** OpenAI chunk stream: role, content, tool calls, finish, usage, [DONE]. */
export function openAIStreamFrames(answer: JsonRecord, dialledId: string): string[] {
  const base: JsonRecord = {
    id: typeof answer.id === "string" ? answer.id : "chatcmpl-answer",
    object: "chat.completion.chunk",
    created: typeof answer.created === "number" ? answer.created : LIST_CREATED,
    model: dialledId,
  };
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const frames: string[] = [];
  const chunk = (delta: JsonRecord, finishReason: unknown): string =>
    sseData({ ...base, choices: [{ index: 0, delta, finish_reason: finishReason }] });

  frames.push(chunk({ role: "assistant", content: "" }, null));
  const text = textOf(message.content);
  if (text !== "") frames.push(chunk({ content: text }, null));
  const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  toolCalls.forEach((rawCall, index) => {
    if (!isRecord(rawCall) || !isRecord(rawCall.function)) return;
    frames.push(
      chunk(
        {
          tool_calls: [
            {
              index,
              id: typeof rawCall.id === "string" ? rawCall.id : `call_${index}`,
              type: "function",
              function: {
                name: typeof rawCall.function.name === "string" ? rawCall.function.name : "",
                arguments: typeof rawCall.function.arguments === "string" ? rawCall.function.arguments : "",
              },
            },
          ],
        },
        null,
      ),
    );
  });
  frames.push(chunk({}, typeof choice.finish_reason === "string" ? choice.finish_reason : "stop"));
  if (isRecord(answer.usage)) {
    frames.push(sseData({ ...base, choices: [], usage: answer.usage }));
  }
  frames.push("data: [DONE]\n\n");
  return frames;
}

/** Anthropic event stream: message_start, blocks, message_delta, stop. */
export function anthropicStreamFrames(answer: JsonRecord, dialledId: string): string[] {
  const translated = anthropicAnswerFromOpenAI(answer, dialledId);
  const content = Array.isArray(translated.content) ? translated.content : [];
  const usage = isRecord(translated.usage) ? translated.usage : { input_tokens: 0, output_tokens: 0 };
  const frames: string[] = [];
  frames.push(
    sseEvent("message_start", {
      type: "message_start",
      message: {
        id: translated.id,
        type: "message",
        role: "assistant",
        model: dialledId,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.input_tokens, output_tokens: 0 },
      },
    }),
  );
  content.forEach((rawBlock, index) => {
    if (!isRecord(rawBlock)) return;
    if (rawBlock.type === "tool_use") {
      frames.push(
        sseEvent("content_block_start", {
          type: "content_block_start",
          index,
          content_block: { type: "tool_use", id: rawBlock.id, name: rawBlock.name, input: {} },
        }),
      );
      frames.push(
        sseEvent("content_block_delta", {
          type: "content_block_delta",
          index,
          delta: { type: "input_json_delta", partial_json: JSON.stringify(rawBlock.input ?? {}) },
        }),
      );
    } else {
      frames.push(
        sseEvent("content_block_start", {
          type: "content_block_start",
          index,
          content_block: { type: "text", text: "" },
        }),
      );
      if (typeof rawBlock.text === "string" && rawBlock.text !== "") {
        frames.push(
          sseEvent("content_block_delta", {
            type: "content_block_delta",
            index,
            delta: { type: "text_delta", text: rawBlock.text },
          }),
        );
      }
    }
    frames.push(sseEvent("content_block_stop", { type: "content_block_stop", index }));
  });
  frames.push(
    sseEvent("message_delta", {
      type: "message_delta",
      delta: { stop_reason: translated.stop_reason, stop_sequence: null },
      usage: { output_tokens: usage.output_tokens },
    }),
  );
  frames.push(sseEvent("message_stop", { type: "message_stop" }));
  return frames;
}

// ---------------------------------------------------------------------
// Model lists and error bodies.
// ---------------------------------------------------------------------

export function openAIModelList(models: FreeModel[]): JsonRecord {
  return {
    object: "list",
    data: models.map((model) => ({
      id: model.id,
      object: "model",
      created: LIST_CREATED,
      owned_by: "openrouter",
    })),
  };
}

export function anthropicModelList(models: FreeModel[]): JsonRecord {
  const list: JsonRecord = {
    data: models.map((model) => ({
      type: "model",
      id: model.id,
      display_name: model.id,
      created_at: LIST_CREATED_AT,
    })),
    has_more: false,
  };
  if (models.length > 0) {
    list.first_id = models[0].id;
    list.last_id = models[models.length - 1].id;
  }
  return list;
}

export function openAIErrorBody(message: string, code: string | null): JsonRecord {
  return { error: { message, type: "upstream_error", param: null, code } };
}

export function anthropicErrorBody(message: string, serverError: boolean): JsonRecord {
  return {
    type: "error",
    error: { type: serverError ? "api_error" : "invalid_request_error", message },
  };
}
