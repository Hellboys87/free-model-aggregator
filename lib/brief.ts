// Reading the trial brief.
//
// The brief (the text in MAKEFUN_PROMPT_FILE) carries two JSON documents
// among its prose: a models document listing every free model with its
// house rating and capabilities, and a prompts document listing the six
// prompts in the order the trial sends them, each with the requirements
// its answer must meet. This module finds those documents, normalises
// them, builds the registration catalog out of the models document, and
// ranks the models for a prompt's requirements.
//
// Everything here is a pure function of the brief text: no clock, no
// randomness, so a replayed trial makes the same choices.

export interface FreeModel {
  /** The dialable id, always carrying the gateway's openrouter/ prefix. */
  id: string;
  /** The house rating in milli-points; higher serves better. */
  ratingMilli: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  contextLength: number | null;
}

export interface Requirements {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

export interface PromptSpec {
  id: string;
  api: "openai" | "anthropic";
  stream: boolean;
  weight: number;
  requires: Requirements;
}

export interface Brief {
  models: FreeModel[];
  prompts: PromptSpec[];
}

type JsonRecord = Record<string, unknown>;

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Find every balanced top-level JSON object in the text that parses and
 * carries a string "schema" field. The brief's prose is not JSON, so a
 * scan that respects string literals and nesting finds exactly the
 * embedded documents.
 */
function schemaDocuments(text: string): JsonRecord[] {
  const documents: JsonRecord[] = [];
  let i = 0;
  while (i < text.length) {
    if (text[i] !== "{") {
      i += 1;
      continue;
    }
    let depth = 0;
    let inString = false;
    let escaped = false;
    let end = -1;
    for (let j = i; j < text.length; j += 1) {
      const ch = text[j];
      if (inString) {
        if (escaped) escaped = false;
        else if (ch === "\\") escaped = true;
        else if (ch === '"') inString = false;
        continue;
      }
      if (ch === '"') inString = true;
      else if (ch === "{") depth += 1;
      else if (ch === "}") {
        depth -= 1;
        if (depth === 0) {
          end = j;
          break;
        }
      }
    }
    if (end === -1) break;
    try {
      const parsed: unknown = JSON.parse(text.slice(i, end + 1));
      if (isRecord(parsed) && typeof parsed.schema === "string") documents.push(parsed);
    } catch {
      // Prose that happens to balance like an object; not a document.
    }
    i = end + 1;
  }
  return documents;
}

function dialableId(row: JsonRecord): string | null {
  if (typeof row.id !== "string" || row.id === "") return null;
  if (row.provider === "openrouter" && !row.id.startsWith("openrouter/")) {
    return `openrouter/${row.id}`;
  }
  return row.id;
}

function toFreeModel(row: JsonRecord): FreeModel | null {
  const id = dialableId(row);
  if (id === null) return null;
  // Capabilities ride on the row itself (tools/vision/json), or nested
  // under a capabilities object in some brief printings.
  const capabilities = isRecord(row.capabilities) ? row.capabilities : row;
  return {
    id,
    ratingMilli: typeof row.rating_milli === "number" ? row.rating_milli : 0,
    tools: capabilities.tools === true,
    vision: capabilities.vision === true,
    json: capabilities.json === true,
    contextLength: typeof row.context_length === "number" ? row.context_length : null,
  };
}

function toRequirements(raw: unknown): Requirements {
  const requires: Requirements = {};
  if (!isRecord(raw)) return requires;
  if (raw.tools === true) requires.tools = true;
  if (raw.vision === true) requires.vision = true;
  if (raw.json === true) requires.json = true;
  if (typeof raw.min_context === "number") requires.minContext = raw.min_context;
  return requires;
}

function toPromptSpec(row: JsonRecord): PromptSpec | null {
  if (typeof row.id !== "string" || row.id === "") return null;
  if (row.api !== "openai" && row.api !== "anthropic") return null;
  return {
    id: row.id,
    api: row.api,
    stream: row.stream === true,
    weight: typeof row.weight === "number" ? row.weight : 1,
    requires: toRequirements(row.requires),
  };
}

/** Parse the brief text into its models and its ordered prompt list. */
export function parseBrief(text: string): Brief {
  const models: FreeModel[] = [];
  const prompts: PromptSpec[] = [];
  for (const doc of schemaDocuments(text)) {
    if (!doc.schema.startsWith("free-model-aggregator-02/")) continue;
    const modelRows = doc.models ?? doc.ratings;
    if (Array.isArray(modelRows)) {
      for (const row of modelRows) {
        if (!isRecord(row) || row.free === false) continue;
        const model = toFreeModel(row);
        if (model !== null && !models.some((m) => m.id === model.id)) models.push(model);
      }
    }
    const promptRows = doc.prompts ?? doc.manifest;
    if (Array.isArray(promptRows)) {
      for (const row of promptRows) {
        if (!isRecord(row)) continue;
        const spec = toPromptSpec(row);
        if (spec !== null) prompts.push(spec);
      }
    }
  }
  return { models, prompts };
}

/**
 * The catalog document registered with the trial environment and printed
 * by --list. One provider, the brief's free models in brief order.
 */
export function buildCatalog(models: FreeModel[]): unknown {
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: "openrouter",
        base_url: "https://openrouter.ai/api/v1",
        key_env: "OPENROUTER_API_KEY",
        wire: "openai",
        models: models.map((model) => ({
          id: model.id,
          free: true,
          context_length: model.contextLength,
        })),
      },
    ],
  };
}

/** True when the model meets every requirement a prompt declares. */
export function meetsRequirements(model: FreeModel, requires: Requirements): boolean {
  if (requires.tools === true && !model.tools) return false;
  if (requires.vision === true && !model.vision) return false;
  if (requires.json === true && !model.json) return false;
  if (typeof requires.minContext === "number") {
    if (model.contextLength === null || model.contextLength < requires.minContext) return false;
  }
  return true;
}

/**
 * The models that may serve a prompt, best first: highest house rating
 * first, ties broken by id so the order never depends on input order.
 * Quality scoring divides the serving model's rating by the best rating
 * among the models meeting the requirements, so this order is the whole
 * strategy: try the best compliant model, then the next.
 */
export function rankCandidates(models: FreeModel[], requires: Requirements): FreeModel[] {
  return models
    .filter((model) => meetsRequirements(model, requires))
    .sort((a, b) => b.ratingMilli - a.ratingMilli || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}
