# free-model-aggregator

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/Hellboys87/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/Hellboys87/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One small Node server that aggregates OpenRouter's free models behind both
client wires at once — OpenAI Chat Completions and Anthropic Messages, JSON
and streaming — and routes every prompt to the strongest free model allowed
to serve it.

## How it works

- **Both wires, one core.** `POST /v1/chat/completions` and
  `POST /v1/messages` are served in JSON and in SSE streaming.
  `GET /v1/models` answers in the wire the auth header names (bearer →
  OpenAI list, `x-api-key` → Anthropic list), listing exactly the brief's
  free models. Anthropic requests are translated to a Chat Completions
  body for the gateway and the answer is translated back; the gateway is
  always dialled non-streaming, and a requested stream is synthesized
  from the complete answer with each wire's event order intact.
- **Selection is the strategy.** The brief lists every free model with a
  house rating and its capabilities (tools, vision, JSON mode, context).
  For each prompt, the candidates meeting *all* of its requirements —
  from the brief's manifest, unioned with what the request body itself
  declares — are ranked by rating, best first. The answer's `model`
  field (and every streamed chunk's) names the model actually dialled,
  with the gateway's `openrouter/` prefix restored, and usage and stop
  reason are carried whole.
- **Failover.** A refusal (429, 5xx, a network error) moves to the
  next-best compliant model, up to four attempts per prompt; the
  refusals a served prompt recovered from are reported in the
  `x-makefun-upstream-failures` header, in refusal order. A model that
  refused once is retried first on the next prompt — a transient rate
  limit should not cost quality for the rest of the trial.
- **Deterministic.** Every choice derives from the brief, the request,
  and the gateway's answers: no wall clock, no randomness, fixed list
  timestamps, response ids taken from the recorded gateway answer. A
  replayed trial repeats the same dials byte for byte.
- **The catalog** is built from the brief at runtime and registered with
  the trial environment; `node main.ts --list` prints the same document.

Layout: `main.ts` (server, routing, failover loop, trial lifecycle),
`lib/brief.ts` (brief parsing, catalog, ranking), `lib/wire.ts` (wire
shapes and translations), `lib/gateway.ts` (the one dial-out point).

## Run it

```sh
node main.ts          # trial server: MAKEFUN_PROMPT_FILE, MAKEFUN_ENV_URL,
                      # OPENAI_BASE_URL and OPENAI_API_KEY come from the trial
node main.ts --list   # print the catalog document and exit
```

Standard library only; Node 22+ runs the TypeScript directly.

---

<sub>Entered on [make.fun](https://make.fun/?ref=9xcq3bkm&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
