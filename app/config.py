"""Centralized configuration for the HotWaterGas AI service.

Uses Pydantic v2 ``BaseSettings`` so values can come from environment
variables or a local ``.env`` file. The class deliberately keeps
generation disabled by default and never logs the resolved values.

Environment variable naming
---------------------------
Pydantic Settings reads variables in this order (first match wins):

1. The exact field name in upper case, e.g. ``APP_ENV``.
2. The configured ``env_prefix`` + field name, e.g. ``HOTWATERGAS_AI_APP_ENV``.

We use the exact-name form (no prefix) to keep parity with the operator
mental model: ``.env.example`` and deployment configs spell the names
out plainly.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


AppEnv = Literal["development", "staging", "production"]

# Provider selector. ``ollama`` is the local, free, offline default used
# during development and tests.
#
# ``gemini`` is still an accepted *value* so an existing deployment's
# ``LLM_PROVIDER`` setting does not become an unparseable value, but the
# Gemini provider itself no longer exists: the legacy pipeline it
# served has been removed. Selecting it therefore fails closed with the
# same "provider is not configured" error as any other unavailable
# backend. There is deliberately **no** automatic fallback between
# providers: if the selected provider is unavailable the service fails
# with a controlled error rather than silently switching models.
LlmProvider = Literal["ollama", "gemini"]


# Default local Ollama endpoint. Ollama is a local process; the default
# binds to the loopback interface only and is never exposed publicly.
DEFAULT_OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"

# Default local model. This is a free, locally-hosted model — no paid
# external API is called during development or tests.
DEFAULT_OLLAMA_MODEL_ID: str = "qwen3:8b"

# Ollama runs locally, so the timeout budget is dominated by CPU
# inference time rather than network latency. ``qwen3:8b`` emits
# reasoning tokens before its answer, so the default is generous.
DEFAULT_OLLAMA_REQUEST_TIMEOUT_SECONDS: int = 120

# Separate, shorter budget for the cheap ``/api/tags`` model-availability
# probe. The probe must never dominate the request latency budget.
DEFAULT_OLLAMA_TAGS_TIMEOUT_SECONDS: int = 10

# ── English description-translation bounds ─────────────────────────────────
#
# The translation path describes a *different* job - translating an
# English Steam description into Vietnamese prose - and its bounds are
# deliberately kept in their own namespaced constants so a future change
# to one cannot silently alter another path's behaviour.
#
# The rules the translation bounds follow:
#
# * The soft target is a **target**, not a requirement. Factual fidelity
#   takes priority over length: when the source does not contain enough
#   material to fill the range, the service returns a shorter faithful
#   translation and an admin-facing warning.
# * The hard ceiling is a **ceiling**. Text above it is rejected, never
#   truncated.
# * The source budget bounds the **English** source the model is shown.
#   It is expressed in UTF-8 bytes because that is the only measure that
#   behaves identically for every language.

#: Lower bound of the soft target range, in whitespace-separated
#: tokens (Vietnamese syllables).
DEFAULT_MIN_TARGET_WORDS: int = 500

#: Upper bound of the soft target range.
DEFAULT_MAX_TARGET_WORDS: int = 600

#: Hard ceiling on the number of whitespace-separated tokens the
#: translated description may contain.
DEFAULT_MAX_DESCRIPTION_WORDS: int = 800

#: Maximum size, in UTF-8 bytes, of the English description source that
#: is shown to the model.
#:
#: The budget is deliberately conservative: the provider input limit
#: is 32 KB, the JSON schema and the system prompt together occupy a few
#: kilobytes, and a 10 KB source leaves a wide margin. The budget is
#: applied at **block boundaries**, so a truncated source is a whole
#: number of paragraphs, headings and list items - never a sentence cut
#: in half.
DEFAULT_MAX_SOURCE_BYTES: int = 10 * 1024

#: Maximum size, in UTF-8 bytes, of the serialised provider input
#: (system prompt plus user prompt plus the JSON schema).
DEFAULT_MAX_PROVIDER_INPUT_BYTES: int = 32768

#: Maximum size, in UTF-8 bytes, of the serialised request. The
#: request carries a product name and an App ID, so it is tiny; the cap
#: exists so the boundary is explicit rather than implicit.
DEFAULT_MAX_REQUEST_BYTES: int = 8192

#: Maximum length of the product name in a request.
DEFAULT_MAX_PRODUCT_NAME_LENGTH: int = 200

#: Minimum number of whitespace-separated tokens the selected English
#: source must carry for the source to be considered **sufficient**.
#:
#: This is the *source* side of the length question, and it is measured
#: on the selected, deduplicated English blocks — never on the model's
#: output. A short translation therefore does **not** imply an
#: insufficient source, and a long translation does not imply a
#: sufficient one.
#:
#: The threshold exists because the target range (500–600 Vietnamese
#: tokens) is set for a *long* store description. It is derived from that
#: range with a documented, deliberately conservative ratio:
#:
#: * Vietnamese is a monosyllabic language: an English passage of *N*
#:   words usually becomes roughly *1.4 × N* Vietnamese tokens, so the
#:   inverse ratio is about *0.71 English tokens per Vietnamese token*.
#: * The threshold applies that inverse ratio to the **lower** bound of
#:   the target range (500), giving about 355 English tokens.
#: * It is then rounded **down** to 350, which is marginally stricter
#:   than the derived value. A source sitting just under the boundary is
#:   therefore reported as insufficient rather than being waved through
#:   on a technicality.
#:
#: The value is a *floor*, not a promise: a source at or above it can
#: still translate to a shorter text (the model may compress), and a
#: source below it may still translate to a perfectly useful short
#: description. The threshold only decides which **warning** is
#: emitted, and it never rejects a faithful short translation.
DEFAULT_MIN_SOURCE_TOKENS: int = 350


#: Legacy environment-variable spellings that are still honoured.
#:
#: The translation bounds were originally spelled with a ``V3_`` prefix
#: in both the constants and the environment variables. The names are
#: now neutral, but a deployment that still exports the old spelling
#: must keep working — silently falling back to the default would
#: quietly change the operator's configured limits.
#:
#: The compatibility is declared on the *fields* themselves, through
#: :class:`pydantic.AliasChoices`: the canonical name is listed first,
#: so it always wins when both are present, and the legacy name is
#: consulted only when the canonical one is absent.
#:
#: This table documents the mapping in one place. It is the list a
#: deployment can read to know which old spellings still work.
LEGACY_ENV_ALIASES: dict[str, str] = {
    "V3_MIN_TARGET_WORDS": "MIN_TARGET_WORDS",
    "V3_MAX_TARGET_WORDS": "MAX_TARGET_WORDS",
    "V3_MAX_DESCRIPTION_WORDS": "MAX_DESCRIPTION_WORDS",
    "V3_MAX_SOURCE_BYTES": "MAX_SOURCE_BYTES",
    "V3_MAX_PROVIDER_INPUT_BYTES": "MAX_PROVIDER_INPUT_BYTES",
    "V3_MIN_SOURCE_TOKENS": "MIN_SOURCE_TOKENS",
}


class Settings(BaseSettings):
    """Typed application settings.

    All fields default to safe values so the service can start in a
    fully-disabled state without any external secrets.
    """

    # ── Application ────────────────────────────────────────────────────
    app_env: AppEnv = Field(
        default="development",
        description=(
            "Runtime environment. One of: development, staging, production. "
            "Affects only logging verbosity in this phase."
        ),
    )

    # ── AI generation switch ───────────────────────────────────────────
    ai_generation_enabled: bool = Field(
        default=False,
        description=(
            "Master switch for AI description generation. Must be true for any "
            "future /internal/descriptions/* route to accept traffic. Defaults "
            "to false so the service starts safely without a provider key."
        ),
    )

    # ── Provider selection ────────────────────────────────────────────
    llm_provider: LlmProvider = Field(
        default="ollama",
        description=(
            "Which LLM backend serves generation requests. 'ollama' is the "
            "local, free, offline default for development and tests. "
            "'gemini' is still accepted as a value so an existing "
            "deployment's setting stays parseable, but that backend no "
            "longer exists and selecting it fails closed. There is NO "
            "automatic fallback between providers — a failure is surfaced "
            "as a controlled error instead of silently switching models."
        ),
    )

    # ── Ollama provider (local) ───────────────────────────────────────
    ollama_base_url: str = Field(
        default=DEFAULT_OLLAMA_BASE_URL,
        description=(
            "Base URL of the local Ollama HTTP API. Must point at a "
            "loopback/private address in every environment; the service "
            "never binds or advertises Ollama publicly."
        ),
    )
    ollama_model_id: str = Field(
        default=DEFAULT_OLLAMA_MODEL_ID,
        description=(
            "Local model id requested from Ollama. Defaults to the free "
            "qwen3:8b model. The service never silently falls back to a "
            "different model."
        ),
    )
    ollama_request_timeout_seconds: int = Field(
        default=DEFAULT_OLLAMA_REQUEST_TIMEOUT_SECONDS,
        ge=1,
        le=600,
        description=(
            "Per-request timeout for a generation call, in seconds. Local "
            "CPU inference is slow, so the default is generous (120s)."
        ),
    )
    ollama_connect_timeout_seconds: int = Field(
        default=5,
        ge=1,
        le=60,
        description=(
            "TCP connection timeout for the local Ollama socket, in "
            "seconds. Kept short so a stopped Ollama server fails fast."
        ),
    )
    ollama_tags_timeout_seconds: int = Field(
        default=DEFAULT_OLLAMA_TAGS_TIMEOUT_SECONDS,
        ge=1,
        le=60,
        description=(
            "Timeout for the /api/tags model-availability probe, in "
            "seconds. The probe is cheap and must not dominate latency."
        ),
    )
    ollama_num_predict: int = Field(
        default=2048,
        ge=1,
        le=32768,
        description=(
            "Upper bound on tokens generated by a single Ollama call. "
            "Sized for a structured draft including reasoning tokens."
        ),
    )
    ollama_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description=(
            "Sampling temperature for Ollama. Kept low so structured "
            "drafts are close to deterministic."
        ),
    )

    # ── Internal service auth ──────────────────────────────────────────
    internal_service_token: str = Field(
        default="",
        description=(
            "Shared secret presented by ASP.NET Core via X-Internal-Token. "
            "The service does not enforce it at startup, "
            "but the value MUST be set in every environment that will later "
            "enable generation. Empty in development is acceptable as long as "
            "ai_generation_enabled is false."
        ),
    )

    # ── Steam retrieval (optional enrichment of the translation) ─────
    steam_retrieval_enabled: bool = Field(
        default=False,
        description=(
            "Master switch for the optional Steam retrieval behind the "
            "translation route. When false a request that carries a "
            "steam_app_id is still accepted, but no Steam request is made. "
            "Defaults to false so the service never reaches out to Steam "
            "unless the operator opts in."
        ),
    )
    steam_connect_timeout_seconds: float = Field(
        default=5.0,
        gt=0,
        le=60,
        description=(
            "TCP connection timeout for the Steam request, in seconds. "
            "Kept short so an unreachable host fails fast."
        ),
    )
    steam_read_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        le=120,
        description=(
            "Read timeout for the Steam request, in seconds. Kept well "
            "below the Ollama budget so Steam can never dominate the "
            "request latency."
        ),
    )
    steam_max_response_bytes: int = Field(
        default=524288,
        ge=1024,
        le=8_388_608,
        description=(
            "Maximum number of bytes read from the Steam endpoint. The "
            "body is counted while it is read, so an oversized response "
            "is aborted instead of buffered."
        ),
    )
    steam_search_max_candidates: int = Field(
        default=15,
        ge=1,
        le=50,
        description=(
            "Maximum number of Steam Store Search candidates returned "
            "to the caller. The Steam search endpoint may report more "
            "items than this; the surplus is discarded rather than "
            "mapped."
        ),
    )
    steam_search_cooldown_seconds: float = Field(
        default=2.0,
        ge=0.0,
        le=60.0,
        description=(
            "Repeated-query cooldown for the Steam Store Search, in "
            "seconds. The window is enforced per process and in memory "
            "only: it is not shared between workers and it does not "
            "survive a restart."
        ),
    )

    # ── Translation bounds ───────────────────────────────────────────
    #
    # These are the translation pipeline's own bounds. They are
    # **separate** from every other path's bounds: a translated
    # description is a document with its own length profile.
    #
    # They are settings fields (rather than compiled-in constants) so a
    # deployment can tighten or relax them without a code change. The
    # defaults are the documented values.
    min_target_words: int = Field(
        default=DEFAULT_MIN_TARGET_WORDS,
        validation_alias=AliasChoices(
            "MIN_TARGET_WORDS",
            "V3_MIN_TARGET_WORDS",
        ),
        ge=1,
        le=10_000,
        description=(
            "Lower bound of the soft target range, in whitespace-"
            "separated tokens. A soft target: a shorter translation is "
            "returned, with a warning, rather than being padded."
        ),
    )
    max_target_words: int = Field(
        default=DEFAULT_MAX_TARGET_WORDS,
        validation_alias=AliasChoices(
            "MAX_TARGET_WORDS",
            "V3_MAX_TARGET_WORDS",
        ),
        ge=1,
        le=10_000,
        description="Upper bound of the soft target range.",
    )
    max_description_words: int = Field(
        default=DEFAULT_MAX_DESCRIPTION_WORDS,
        validation_alias=AliasChoices(
            "MAX_DESCRIPTION_WORDS",
            "V3_MAX_DESCRIPTION_WORDS",
        ),
        ge=1,
        le=100_000,
        description=(
            "Hard upper bound on the description's token count. A "
            "translation above it is rejected, never truncated."
        ),
    )
    max_source_bytes: int = Field(
        default=DEFAULT_MAX_SOURCE_BYTES,
        validation_alias=AliasChoices(
            "MAX_SOURCE_BYTES",
            "V3_MAX_SOURCE_BYTES",
        ),
        ge=1024,
        le=1_048_576,
        description=(
            "Maximum size, in UTF-8 bytes, of the English description "
            "source shown to the model. Applied at whole-block "
            "boundaries."
        ),
    )
    max_provider_input_bytes: int = Field(
        default=DEFAULT_MAX_PROVIDER_INPUT_BYTES,
        validation_alias=AliasChoices(
            "MAX_PROVIDER_INPUT_BYTES",
            "V3_MAX_PROVIDER_INPUT_BYTES",
        ),
        ge=1024,
        le=4_194_304,
        description=(
            "Maximum size, in UTF-8 bytes, of the serialised provider "
            "input (system prompt plus user prompt plus the JSON schema)."
        ),
    )
    min_source_tokens: int = Field(
        default=DEFAULT_MIN_SOURCE_TOKENS,
        validation_alias=AliasChoices(
            "MIN_SOURCE_TOKENS",
            "V3_MIN_SOURCE_TOKENS",
        ),
        ge=1,
        le=100_000,
        description=(
            "Minimum number of whitespace-separated tokens the selected "
            "English source must carry for the source to be considered "
            "sufficient. Measured on the source, never on the model's "
            "output. It only selects which warning is emitted; it never "
            "rejects a faithful short translation."
        ),
    )

    # ── Translation bounds ───────────────────────────────────────────
    #
    # These are the translation pipeline's own bounds. They are
    # **separate** from every other path's bounds: a translated
    # description is a longer, document with its own length profile.
    #
    # The bounds themselves are declared as settings fields above (so a
    # deployment can tune them); the constants in :mod:`app.config` are
    # their documented defaults.

    @property
    def max_request_bytes(self) -> int:
        """Maximum size in bytes of the serialised request."""
        return DEFAULT_MAX_REQUEST_BYTES

    @property
    def max_product_name_length(self) -> int:
        """Maximum length of the product name in a request."""
        return DEFAULT_MAX_PRODUCT_NAME_LENGTH

    # ── Pydantic settings configuration ────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # Read raw values for booleans so "true"/"false"/"1"/"0" all work.
        # Pydantic v2 already handles common boolean spellings.
    )

    @field_validator("ai_generation_enabled", mode="before")
    @classmethod
    def _coerce_bool(cls, value: object) -> object:
        """Coerce common boolean spellings before Pydantic's strict check.

        Accepts the strings ``"true"``/``"false"`` (any case), ``"1"``/``"0"``,
        and ``"yes"``/``"no"`` in addition to native booleans. Unknown values
        raise a normal Pydantic validation error.
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off", ""}:
                return False
        return value  # Let Pydantic raise the validation error.

    # ── Derived helpers ────────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    def __repr__(self) -> str:  # pragma: no cover - defensive
        # Never include secret material in repr; this protects logs that
        # accidentally dump the Settings object.
        return (
            "Settings("
            f"app_env={self.app_env!r}, "
            f"ai_generation_enabled={self.ai_generation_enabled}, "
            f"llm_provider={self.llm_provider!r}, "
            f"ollama_base_url={self.ollama_base_url!r}, "
            f"ollama_model_id={self.ollama_model_id!r}, "
            f"ollama_request_timeout_seconds={self.ollama_request_timeout_seconds}, "
            "internal_service_token='***'"
            ")"
        )


def get_settings() -> Settings:
    """FastAPI dependency that returns a freshly-built ``Settings`` instance.

    The instance is created on every request so test fixtures that mutate
    environment variables are picked up without re-importing the module.
    There is no expensive initialisation, so this is cheap.
    """

    return Settings()
