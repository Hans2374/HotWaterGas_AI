# HotWaterGas AI Service

A small, isolated [FastAPI](https://fastapi.tiangolo.com/) service that hosts
the AI Product Description Generation feature for the HotWaterGas e-commerce
platform.

The service exposes three routes:

| Route | Purpose |
| --- | --- |
| `GET /health` | Liveness check. |
| `POST /internal/descriptions/generate` | English Steam description -> Vietnamese translation. |
| `POST /internal/steam/search` | Steam Store Search. |

Every generation route is served by the **local, free Ollama provider**.
The legacy Gemini-backed pipeline and the structured-draft pipeline have
been removed, along with their routes (including
`POST /internal/descriptions/generate-v2`): this service contacts no
hosted or paid LLM backend.

`POST /internal/descriptions/generate-v3` is still accepted as a
temporary, backward-compatible alias of the canonical route. It is
hidden from the OpenAPI schema; new clients should use the canonical
path.

The intended future architecture is:

```
React Admin UI  ->  ASP.NET Core 8 API  ->  Python FastAPI (this service)  ->  local Ollama
                                                                    +->  public Steam metadata (optional)
```

ASP.NET Core remains the source of truth for authentication, authorization,
product data, and persistence. Python handles AI-specific logic only.

> **Naming note.** The implementation is consolidated under neutral names.
> `SCHEMA_VERSION` stays `"3"` and the prompt version stays `"v3.1"` on
> the wire, so the response payload is unchanged. The legacy
> `POST /internal/descriptions/generate-v3` path is a temporary alias of
> the canonical route and is hidden from the OpenAPI schema.

---

## Requirements

* **Python 3.12** (developed and tested on `3.12.10`).
* **PowerShell** on Windows (the commands below are PowerShell-friendly; the
  `py` launcher and `python` interpreter both work).
* `pip` 25.x or newer.
* No global packages are required; everything installs into a project-local
  virtual environment.
* **Ollama** for the local provider. It is free and offline
  — no paid API is called during development or in the test suite.

> Do **not** install FastAPI or Pydantic globally. Use a virtual environment
> as shown below.

---

## Local setup (Windows PowerShell)

From the repository root:

```powershell
# 1. Create a project-local virtual environment
python -m venv HotWaterGas_AI\.venv

# 2. Activate it
HotWaterGas_AI\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install -r HotWaterGas_AI\requirements.txt

# 4. Create a local .env from the example (optional; the defaults are safe)
Copy-Item HotWaterGas_AI\.env.example HotWaterGas_AI\.env
```

To leave the virtual environment when you are done:

```powershell
deactivate
```

### Ollama

The generation endpoints use a local Ollama server. Install it once, then pull the
model:

```powershell
ollama --version          # expect 0.40.x or newer
ollama pull qwen3:8b
ollama serve              # starts the API on 127.0.0.1:11434
```

Verify the server and the model are reachable:

```powershell
Invoke-RestMethod http://127.0.0.1:11434/api/tags | ConvertTo-Json -Depth 3
```

> Ollama must stay bound to **loopback**. Never expose it on a public
> interface, and never point `OLLAMA_BASE_URL` at a public host.

---

## Configuration

Configuration is loaded by **Pydantic Settings** from environment variables
and (optionally) a local `.env` file. See `app/config.py` for the full list.

| Variable | Default | Notes |
| --- | --- | --- |
| `APP_ENV` | `development` | One of `development`, `staging`, `production`. |
| `AI_GENERATION_ENABLED` | `false` | **Master switch for description generation.** When `false`, every generation route returns HTTP 503 with `code: generation_disabled`. |
| `LLM_PROVIDER` | `ollama` | Which backend serves generation. `ollama` (local, free) is the only implemented backend. `gemini` is still accepted as a value so an existing deployment's setting stays parseable, but that backend was removed and selecting it fails closed. There is **no automatic fallback** between providers. |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Local Ollama HTTP API. Keep it on loopback; never point it at a public host. |
| `OLLAMA_MODEL_ID` | `qwen3:8b` | The local model. The service never silently falls back to a different model. |
| `OLLAMA_REQUEST_TIMEOUT_SECONDS` | `120` | Per-request timeout, 1–600 s. Local CPU inference is slow, so the default is generous. |
| `OLLAMA_CONNECT_TIMEOUT_SECONDS` | `5` | TCP connect timeout, 1–60 s. Short, so a stopped Ollama fails fast. |
| `OLLAMA_TAGS_TIMEOUT_SECONDS` | `10` | Timeout for the `/api/tags` model-availability probe, 1–60 s. The result is cached. |
| `OLLAMA_NUM_PREDICT` | `2048` | Upper bound on generated tokens per call. |
| `OLLAMA_TEMPERATURE` | `0.2` | Sampling temperature, 0.0–2.0. Kept low for near-deterministic drafts. |
| `STEAM_RETRIEVAL_ENABLED` | `false` | Master switch for the optional Steam retrieval behind the translation route. When `false`, the route answers `502 steam_unavailable` rather than reaching out to Steam. |
| `INTERNAL_SERVICE_TOKEN` | `""` | Shared secret that ASP.NET Core will present via the `X-Internal-Token` header. When empty the service fails closed: any `/internal/*` request returns HTTP 500 with `code: internal_token_not_configured`. Token values are restricted to ASCII (RFC 7230). Generate with `python -c "import secrets; print(secrets.token_urlsafe(48))"`. |

> The `AI_GENERATION_ENABLED` flag accepts `true`, `false`, `1`, `0`, `yes`,
> `no`, `on`, `off`, and the empty string (treated as `false`). Unknown
> values raise a Pydantic validation error at startup.

### Compile-time bounds (not env-driven)

The following bounds are constants in `app/config.py`. They are exposed on
the `Settings` object so application code reads a single typed surface:

### Translation bounds

These bound what the translation endpoint accepts and produces. They are
compile-time constants rather than environment settings for the request
boundaries: they are part of the request contract, so an operator cannot
relax them and silently change what the API accepts.

| Bound | Value | Where it is enforced |
| --- | --- | --- |
| `min_target_words` | `500` | Soft target range. A shorter translation is returned with a warning, never padded. |
| `max_target_words` | `600` | Soft target range. |
| `max_description_words` | `800` | Hard ceiling. Text above it is rejected, never truncated. |
| `max_source_bytes` | `10240` | The English source shown to the model. Applied at whole-block boundaries. |
| `max_provider_input_bytes` | `32768` | Service boundary, before the model is called. HTTP 413 `request_too_large`. |
| `max_request_bytes` | `8192` | Service boundary, before retrieval. HTTP 413 `request_too_large`. |
| `max_product_name_length` | `200` | Request schema. A whitespace-only name is rejected. |
| `min_source_tokens` | `350` | The source-sufficiency threshold. It selects the warning, never rejects. |

Every one of these **rejects**; none of them truncates. A request that is
refused tells the caller that something was wrong, while a request that is
trimmed succeeds while quietly shipping a different product than the one
that was asked about.

The two aggregate bounds count UTF-8 **bytes**, not characters and not
tokens. A character limit would behave differently for Vietnamese than for
ASCII, and a token limit would pretend the service owns the model's
tokeniser. A Vietnamese request costs roughly 1.4 bytes per character, so
the byte bound is slightly stricter for Vietnamese and exactly the same for
ASCII.

---

## Running the service locally

From the repository root, with the virtual environment activated:

```powershell
# Default host/port: 127.0.0.1:8000
python -m uvicorn app.main:app --reload --port 8000
```

Once the server is up, verify the liveness endpoint:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json
```

The response is a small JSON document:

```json
{
  "status": "ok",
  "service": "hotwatergas-ai",
  "version": "0.1.0",
  "ai_generation_enabled": false
}
```

The interactive API documentation is available at:

* Swagger UI — <http://127.0.0.1:8000/docs>
* ReDoc — <http://127.0.0.1:8000/redoc>

---

## Endpoints

### `GET /health` — liveness

A **liveness** check. It confirms only that the Python process and the
FastAPI application are running. It does **not**:

* Verify the provider's connectivity.
* Check the validity of the local model's configuration.
* Verify the `INTERNAL_SERVICE_TOKEN`.
* Confirm that AI generation is operationally ready.

A future **readiness** check may be added in a later phase.

### `POST /internal/descriptions/generate` — English → Vietnamese translation

Translates the **English** Steam store description of an asserted App ID into
Vietnamese prose, through the local Ollama provider. It is not part of the
public API and requires the `X-Internal-Token` header.

The request carries a product name and an App ID:

```bash
curl -X POST http://127.0.0.1:8000/internal/descriptions/generate \
  -H "Content-Type: application/json" \
  -H "X-Internal-Token: $TOKEN" \
  -d '{
    "product_name": "Echoes of the Tundra",
    "steam_app_id": 620
  }'
```

**What the route returns**

| Area | Meaning |
| --- | --- |
| `schema_version` | Always `"3"`. |
| `description` | The Vietnamese text, its server-measured token count, and its verification state. |
| `release_date` | ISO-8601 `YYYY-MM-DD` or `null`. Non-ISO spellings are dropped. |
| `platforms` | A list of platform names, or an empty list when unknown. |
| `system_requirements.minimum` | The five PC fields `os`, `processor`, `memory`, `graphics`, `storage`. |
| `system_requirements.recommended` | The same five fields, **fully independent** of `minimum`. |
| `description_source` | **Always present.** Describes the English source the model was shown. |
| `warnings` | Admin-facing warnings, each with a stable `code`. |
| `evidence` | The records the service minted from the retrieved Steam payload. |
| `model` | Provider, model id, prompt version, duration. |

**The translated description is never `verified`**

A translation that completed is not a fact that was checked: the service
translates, it does not compare the translation against the source. The
contract enforces this — a payload that claims `verified` is rejected
outright rather than being quietly downgraded, and the status is always
`unverified`.

**The `description_source` block**

The block is **service-owned**: it describes the *input* the model was shown,
not anything the model produced. It names the Steam field the primary blocks
came from, the supplementary fields that contributed, the fields that were
consulted but carried nothing, the block count, the measured token count, and
whether the source was considered sufficient.

**The `warnings` collection**

Every warning carries a **stable code** and a severity. A consumer should
branch on `code`, never on the message.

| Code | Meaning |
| --- | --- |
| `source_truncated` | The source was cut at a whole-block boundary to fit the source budget. |
| `source_insufficient` | The selected source carried too little material to support the target length. |
| `translation_below_target` | The translation came out below the soft target range. It is faithful and unpadded. |
| `description_over_ceiling` | The translation exceeded the hard ceiling and the call failed. |
| `source_block_dropped` | At least one source block did not fit inside the source budget. |

There is deliberately **no** `source_missing` code: the no-description path is
a controlled HTTP error, not a successful response, so a warning code for it
could never appear in a delivered body.

**Optional Steam retrieval**

The retrieval only happens when `STEAM_RETRIEVAL_ENABLED=true` (the default is
`false`, so the service never reaches out to Steam unless an operator opts
in). The switch affects the translation route only.

When the retrieved title does not match the supplied product name, the call
fails with **HTTP 409** (`steam_identity_mismatch`): the retrieved facts
describe a *different* product, and attaching them would be worse than
failing.

**Error responses**

| HTTP | `code` | When |
| --- | --- | --- |
| 401 | `unauthorized` | The internal token is missing or invalid. |
| 404 | `steam_not_found` | The App ID identifies no retrievable Steam app. |
| 409 | `steam_identity_mismatch` | The retrieved title differs from the supplied product name. |
| 413 | `request_too_large` | The request or the assembled provider input exceeds the configured size limit. |
| 422 | `invalid_request` | The request payload is invalid. |
| 500 | `service_not_initialised` | The lifespan handler could not initialise the service. |
| 500 | `internal_token_not_configured` | The server has no internal token configured. |
| 502 | `provider_model_not_found` | The configured local model is not installed in Ollama. |
| 502 | `provider_unavailable` | Ollama is unreachable, timed out, or returned an error. |
| 502 | `invalid_provider_output` | Ollama returned malformed JSON or a payload that violates the schema. |
| 502 | `steam_unavailable` | Steam could not be reached, or published no translatable description. |
| 503 | `generation_disabled` | `AI_GENERATION_ENABLED` is `false`. |
| 503 | `provider_not_configured` | The selected provider is not configured for structured generation. |
| 504 | `provider_timeout` | Ollama did not respond before the configured timeout. |

> **No automatic regeneration.** The service calls the model exactly once per
> request. A translation that comes out short is reported with a warning and
> returned as-is; it is never padded, and it is never regenerated.

### `POST /internal/steam/search` — Steam Store Search

Looks up candidate Steam apps for a game title. It is not part of the public
API and requires the `X-Internal-Token` header.

```bash
curl -X POST http://127.0.0.1:8000/internal/steam/search \
  -H "Content-Type: application/json" \
  -H "X-Internal-Token: $TOKEN" \
  -d '{ "query": "Portal 2" }'
```

**A search result is a candidate, not a fact.** The endpoint exposes only what
a caller needs to *pick* a candidate:

| Field | Meaning |
| --- | --- |
| `app_id` | The Steam App ID the `appdetails` endpoint accepts. |
| `name` | The candidate's title as Steam reports it. |
| `type` | Steam's raw type string. Informational only — it does not distinguish a full game from a demo or a soundtrack. |
| `tiny_image` | The capsule image URL, when Steam reports one. |

The price block, the metascore, the platform flags and the streaming-video
flag are deliberately **not** exposed. They are product facts, and a search
result is not a source of product facts.

The response also carries `total` (how many matches Steam reported — it may
exceed the number of returned candidates), `candidate_count`, `invalid_count`
(items discarded for carrying no usable App ID or title) and `truncated_count`
(valid candidates dropped because the candidate cap had been reached).

The route is disabled when `STEAM_RETRIEVAL_ENABLED=false`, and the same
query is rate-limited by a short in-memory cooldown.

**Error responses**

| HTTP | `code` | When |
| --- | --- | --- |
| 401 | `unauthorized` | The internal token is missing or invalid. |
| 422 | `invalid_request` | The query is missing, empty or too long. |
| 429 | `steam_search_cooldown` | The same query was searched too recently. |
| 429 | `steam_rate_limited` | Steam itself is rate-limiting the request. |
| 500 | `service_not_initialised` | The lifespan handler could not initialise the service. |
| 500 | `internal_token_not_configured` | The server has no internal token configured. |
| 502 | `steam_search_unavailable` | Steam could not be reached or returned an unusable response. |
| 503 | `steam_search_disabled` | `STEAM_RETRIEVAL_ENABLED` is `false`. |

---

## Running the tests

```powershell
# From the repository root, with the virtual environment activated
python -m pytest HotWaterGas_AI
```

Pytest is configured by `HotWaterGas_AI/pytest.ini`. Tests live under
`HotWaterGas_AI/tests/`. **No network access, no provider credential, no paid
API, no live Ollama server and no database are required.** Every test that
needs a provider builds a fake or mocks the transport; the real Ollama
server is never called.

The current test suite covers:

* `test_config.py` — default values, boolean coercion, repr safety, the removal of the structured-draft settings, and the legacy environment-variable aliases.
* `test_config_provider.py` — provider selection, Ollama defaults, timeout bounds, repr safety.
* `test_health.py` — `/health` status, JSON shape, no secret leakage.
* `test_startup.py` — the FastAPI app boots without a provider credential.
* `test_description_fields.py` — the version-neutral shared field value objects and their pinned schema identities.
* `test_validators.py` — output text normalisation, word counting, hard ceiling.
* `test_auth.py` — internal-token dependency, constant-time comparison, ASCII-only.
* `test_errors.py` — error class → HTTP status / public code mapping.
* `test_errors_http.py` — the shared HTTP error helpers and auth handlers.
* `test_ollama_provider.py` — Ollama request construction, structured parsing, error mapping, model check, sanitised errors.
* `test_steam_retriever.py` — the Steam HTTP boundary: timeouts, content types, size caps, no retry.
* `test_steam_evidence_mapping.py` — the mapping from a Steam payload to facts and evidence.
* `test_steam_deterministic_facts.py` — the deterministic facts derived from a Steam payload.
* `test_steam_search.py` — the Steam Store Search service layer.
* `test_steam_search_endpoint.py` — the Steam Store Search route, its auth and its contract.
* `test_description_pipeline.py` — the translation pipeline and its contract.
* `test_offline_guards.py` — self-checks on the egress and dotenv guards that keep the suite offline.

---

## Manual smoke test — local Ollama

> This is a **local, free** call. It requires a running Ollama server with
> `qwen3:8b` pulled. It is **not** part of the automated test suite.

### 1. Start Ollama and confirm the model

```powershell
ollama serve
Invoke-RestMethod http://127.0.0.1:11434/api/tags | ConvertTo-Json -Depth 3
```

### 2. Configure the service

```ini
APP_ENV=development
AI_GENERATION_ENABLED=true
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL_ID=qwen3:8b
OLLAMA_REQUEST_TIMEOUT_SECONDS=120
INTERNAL_SERVICE_TOKEN=a-strong-ascii-secret
```

### 3. Start the service

```powershell
python -m uvicorn app.main:app --port 8000
```

### 4. Request a translation

Use a fictional, non-sensitive product:

```powershell
$token = "a-strong-ascii-secret"
$body = @'
{
  "product_name": "Echoes of the Tundra",
  "steam_app_id": 620
}
'@
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/internal/descriptions/generate" `
  -ContentType "application/json" `
  -Headers @{ "X-Internal-Token" = $token } `
  -Body $body
```

### 5. Verify the response

* HTTP 200 with `schema_version == "3"`.
* `description.verification.status` is `unverified`. The service
  translates; it never claims the translation was checked.
* `description.word_count` is the server's own count of the text it
  shipped, and `within_target_range` says whether it sits inside the
  soft target range.
* `description_source` is present, and it describes the **source**:
  `primary_field`, `block_count`, `source_bytes` and `source_sufficient`
  are properties of the English text, not of the translation.
* `warnings` is present. A short translation produces a
  `translation_below_target` warning; the text is returned as-is and is
  never padded.
* `system_requirements.minimum` and `.recommended` are independent, and
  only the **PC** tier is mapped: Steam's macOS and Linux requirements
  are never read.
* The response body contains neither the internal token nor any price,
  key, or payment information.

### Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| HTTP 502 with `code: provider_unavailable` | Ollama is not running, or `OLLAMA_BASE_URL` is wrong. Check `http://127.0.0.1:11434/api/tags`. |
| HTTP 502 with `code: provider_model_not_found` | `qwen3:8b` is not pulled. Run `ollama pull qwen3:8b`. |
| HTTP 504 with `code: provider_timeout` | Local inference exceeded `OLLAMA_REQUEST_TIMEOUT_SECONDS`. Raise it, or use a smaller model. |
| HTTP 502 with `code: invalid_provider_output` | The model returned malformed JSON or violated the schema. The raw content is not logged; check the server log for the `schema_violation` warning. |
| HTTP 503 with `code: provider_not_configured` | `LLM_PROVIDER` is not `ollama`, or `OLLAMA_MODEL_ID` is empty. |

---

## Cost, quotas, and rate limits

The service applies conservative input and output bounds appropriate for
short product descriptions. The current defaults are:

* Max tokens per translation: **800**.
* Per-request timeout: **120 seconds** (local Ollama).
* Max request payload: **8 192 bytes**.
* Max provider input: **32 768 bytes**.

These are application-level safety nets. **They are not a billing limit.**
The generation path is served by a **local** Ollama model, so it incurs
no per-token cost and calls no external or paid API. The local provider
maps every upstream status of 400 or above to a `502 provider_unavailable`,
so an upstream rate limit surfaces as a 502 and never as a 429.

## What is intentionally NOT implemented yet

* No integration with the React frontend or the ASP.NET Core backend for the
  translation route.
* No **name-based disambiguation** of a Steam App ID. When the retrieved
  title does not match the requested product, the call fails with HTTP 409
  rather than searching for the right app; explicit disambiguation is a
  separate design task.
* No **second LLM verifier** and no semantic verification. The translation
  path never marks a translation `verified`: a translation that completed
  is not a fact that was checked.
* No **correction retries**. The service never regenerates: one provider
  call per request.
* No multi-agent systems, no RAG, no vector databases.
* No automatic saving or publishing of generated descriptions.
* No regeneration loops. One provider call per request.
* No Docker image (will be added when the deployment host is chosen).
* No CI workflow (will be added when one is selected).
* No production deployment configuration.

These are deliberate omissions and will be addressed in later phases per the
project feature checklist.

---

## Project layout

```
HotWaterGas_AI/
├── .env.example
├── .gitignore
├── README.md
├── pytest.ini
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── config.py                 # Pydantic Settings, provider selection, bounds
│   ├── errors.py                 # Domain exceptions → HTTP status mapping
│   ├── errors_http.py            # shared HTTP error helpers + auth handlers
│   ├── logging_config.py         # Safe root-logger configuration
│   ├── main.py                   # FastAPI app factory + lifespan
│   ├── providers/
│   │   ├── __init__.py           # provider factory (LLM_PROVIDER selection)
│   │   ├── base.py               # StructuredProvider protocol
│   │   └── ollama.py             # local Ollama wrapper, structured output
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── steam.py              # Steam appdetails HTTP boundary (optional)
│   │   ├── steam_mapper.py       # Steam payload → facts + evidence
│   │   ├── steam_models.py       # retrieval result types
│   │   ├── steam_search.py        # Steam Store Search HTTP boundary
│   │   ├── steam_search_mapper.py # Steam search payload → search results
│   │   ├── steam_search_models.py # Steam Search result types
│   │   └── description_source.py # description source: HTML → blocks, dedup, budget
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── description_fields.py # version-neutral shared field value objects
│   │   ├── description.py       # translation contract
│   │   └── steam_search.py       # Steam Search request/response models
│   ├── validators/
│   │   ├── __init__.py
│   │   └── output.py             # deterministic text validation
│   ├── services/
│   │   ├── __init__.py
│   │   ├── prompts.py            # translation-only prompt assembly
│   │   └── generation.py         # English→Vietnamese translation orchestrator
│   ├── dependencies/
│   │   ├── __init__.py
│   │   └── auth.py               # require_internal_token dependency
│   └── routers/
│       ├── __init__.py
│       ├── health.py             # GET /health
│       ├── descriptions.py       # POST /internal/descriptions/generate
│       └── steam_search.py       # POST /internal/steam/search
└── tests/
    ├── __init__.py
    ├── conftest.py               # fixtures + offline egress / dotenv guards
    ├── fixtures/steam/           # canned Steam payloads (no network)
    │   └── README.md             # fixture provenance (source URL, locale, date)
    ├── test_config.py
    ├── test_health.py
    ├── test_startup.py
    ├── test_description_fields.py
    ├── test_validators.py
    ├── test_auth.py
    ├── test_errors.py
    ├── test_errors_http.py
    ├── test_ollama_provider.py
    ├── test_steam_retriever.py
    ├── test_steam_evidence_mapping.py
    ├── test_steam_deterministic_facts.py
    ├── test_steam_search.py
    ├── test_steam_search_endpoint.py
    ├── test_description_pipeline.py
    └── test_offline_guards.py
```
