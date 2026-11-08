"""Domain-level exceptions for the HotWaterGas AI service.

The route layer maps these exceptions to documented HTTP status codes; the
service and provider layers raise them and never log credentials, raw SDK
responses, or stack traces. Generic, sanitised messages are returned to the
client. Detailed exception context is logged inside the application
boundary.
"""

from __future__ import annotations


class HotWaterGasAIError(Exception):
    """Base class for all expected, in-band errors.

    Subclasses carry enough metadata for the route layer to map them to a
    stable HTTP status code, but they never include credentials or SDK
    responses in their string representation.
    """

    http_status: int = 500
    public_code: str = "internal_error"

    def __init__(self, message: str = "") -> None:
        super().__init__(message or self.__class__.__name__)
        self.message = message or self.__class__.__name__


# ── Configuration / access control ─────────────────────────────────────────


class GenerationDisabledError(HotWaterGasAIError):
    """Raised when AI generation is requested but the master switch is off.

    HTTP 503 — the client should not retry until the operator enables
    generation in the server-side configuration.
    """

    http_status = 503
    public_code = "generation_disabled"


class MissingInternalTokenConfigError(HotWaterGasAIError):
    """Raised when the server has no internal token configured.

    This is a server-side misconfiguration: the service cannot accept any
    internal requests safely. Returned as HTTP 500 so the operator notices.
    The public message intentionally hides the exact reason.
    """

    http_status = 500
    public_code = "internal_token_not_configured"


class InvalidInternalTokenError(HotWaterGasAIError):
    """Raised when the caller does not present a valid internal token.

    HTTP 401 — the request is unauthorised. No information about the
    expected token is ever returned to the client.
    """

    http_status = 401
    public_code = "unauthorized"


class MissingGeminiConfigError(HotWaterGasAIError):
    """Raised when generation is enabled but required provider settings are absent.

    HTTP 503 — the client should not retry until the operator provides
    credentials. The public message hides which credential is missing.

    The class name is retained for backward compatibility with the .NET
    client's error mapping. The Gemini backend itself has been removed;
    the local Ollama backend raises the sibling
    :class:`MissingOllamaConfigError`, which maps to the same public
    code.
    """

    http_status = 503
    public_code = "provider_not_configured"


class MissingOllamaConfigError(HotWaterGasAIError):
    """Raised when generation is enabled but the Ollama settings are absent.

    HTTP 503 — the same public code as
    :class:`MissingGeminiConfigError` so the route layer and the .NET
    client keep a single "provider is not configured" contract
    regardless of which provider is selected.
    """

    http_status = 503
    public_code = "provider_not_configured"


# ── Provider / upstream ────────────────────────────────────────────────────


class ProviderError(HotWaterGasAIError):
    """Base class for errors reported by the upstream provider.

    Subclasses translate Google's gRPC-style status codes into our HTTP
    mapping. The string representation never includes the raw response body
    or the API key.
    """

    http_status = 502
    public_code = "provider_unavailable"


class ProviderAuthError(ProviderError):
    """The upstream rejected the configured API key or model id."""

    http_status = 502
    public_code = "provider_auth_failed"


class ProviderModelNotFoundError(ProviderError):
    """The configured model id is not recognised by the upstream API."""

    http_status = 502
    public_code = "provider_model_not_found"


class ProviderRateLimitError(ProviderError):
    """The upstream is rate-limiting our requests.

    Mapped to HTTP 429 so callers can implement back-off. The response body
    does not carry retry hints, only the status code.
    """

    http_status = 429
    public_code = "rate_limited"


class ProviderTimeoutError(ProviderError):
    """The upstream took too long to respond."""

    http_status = 504
    public_code = "provider_timeout"


class ProviderUnavailableError(ProviderError):
    """The upstream is currently unavailable (5xx, network, etc.)."""

    http_status = 502
    public_code = "provider_unavailable"


# ── Output / request validation ────────────────────────────────────────────


class InvalidProviderOutputError(ProviderError):
    """The upstream returned a response that cannot be turned into a description.

    Covers empty text, safety blocks, response truncation beyond the
    configured ceiling, and any other case where the upstream did produce
    a response but the result is unusable.
    """

    http_status = 502
    public_code = "invalid_provider_output"


class RequestTooLargeError(HotWaterGasAIError):
    """The aggregated request body is larger than the configured input cap."""

    http_status = 413
    public_code = "request_too_large"


# ── Steam retrieval ───────────────────────────────────────────────────────
#
# Steam retrieval is an **optional enrichment** of the draft. A Steam
# failure must never fabricate a fact, and it must never take the whole
# generation call down: the service degrades to a partial draft with the
# Steam-dependent fields left ``unknown``.
#
# The errors below therefore carry a *diagnosis* (which is safe to log)
# rather than a raw upstream message. The public message is canned by the
# route layer, exactly like every other domain error.


class SteamRetrievalError(HotWaterGasAIError):
    """Steam could not be reached or returned an unusable response.

    Covers connection failures, timeouts, non-JSON content types,
    malformed payloads and oversized responses. The instance carries a
    short ``outcome`` code that the service logs; it never carries the
    upstream response body.
    """

    http_status = 502
    public_code = "steam_unavailable"

    def __init__(self, message: str = "", *, outcome: str = "unknown") -> None:
        super().__init__(message or self.__class__.__name__)
        self.outcome = outcome


class SteamNotFoundError(HotWaterGasAIError):
    """The supplied App ID does not identify a retrievable Steam app.

    Raised when Steam answers ``success: false`` or when the response
    carries no ``data`` object. This is distinct from a transport
    failure: the App ID itself appears to be wrong.
    """

    http_status = 404
    public_code = "steam_not_found"

    def __init__(self, message: str = "", *, outcome: str = "not_found") -> None:
        super().__init__(message or self.__class__.__name__)
        self.outcome = outcome


class SteamIdentityMismatchError(HotWaterGasAIError):
    """The retrieved game title does not match the supplied product name.

    The App ID was asserted by the caller but the retrieved title is
    materially different, so the retrieved facts describe a *different*
    product. Using them would silently attach the wrong requirements to
    the wrong game, so the call fails with a safe, sanitized outcome
    instead of an enriched draft.

    No alternative App ID is selected and no name-search endpoint is
    queried: explicit disambiguation is a separate, future design task.
    """

    http_status = 409
    public_code = "steam_identity_mismatch"

    def __init__(self, message: str = "", *, outcome: str = "identity_mismatch") -> None:
        super().__init__(message or self.__class__.__name__)
        self.outcome = outcome


class SteamSearchDisabledError(HotWaterGasAIError):
    """Steam search was requested but the feature switch is off.

    HTTP 503 - the operator has switched Steam retrieval off, so no
    request is made. This is a server-side configuration state, not a
    client error.
    """

    http_status = 503
    public_code = "steam_search_disabled"

    def __init__(self, message: str = "", *, outcome: str = "disabled") -> None:
        super().__init__(message or self.__class__.__name__)
        self.outcome = outcome


class SteamSearchRateLimitError(HotWaterGasAIError):
    """Steam itself is rate-limiting the search endpoint.

    HTTP 429 - the caller should back off before retrying. This is
    distinct from :class:`SteamSearchCooldownError`, which is the
    service's own repeated-query guard.
    """

    http_status = 429
    public_code = "steam_rate_limited"

    def __init__(self, message: str = "", *, outcome: str = "rate_limited") -> None:
        super().__init__(message or self.__class__.__name__)
        self.outcome = outcome


class SteamSearchUnavailableError(HotWaterGasAIError):
    """Steam could not be reached or returned an unusable response.

    HTTP 502 - the upstream failed. The public message is canned and
    the upstream response body is never included.
    """

    http_status = 502
    public_code = "steam_search_unavailable"

    def __init__(self, message: str = "", *, outcome: str = "unknown") -> None:
        super().__init__(message or self.__class__.__name__)
        self.outcome = outcome


class SteamSearchQueryInvalidError(HotWaterGasAIError):
    """The supplied Steam search term is not usable.

    This is a caller error (HTTP 422): the term was empty after
    trimming, or longer than the allowed maximum. The diagnostic detail
    stays in the logs - the term itself is never echoed back, because
    it is user-supplied free text.
    """

    http_status = 422
    public_code = "invalid_steam_search_query"

    def __init__(self, message: str = "", *, outcome: str = "invalid_query") -> None:
        super().__init__(message or self.__class__.__name__)
        self.outcome = outcome


class SteamSearchCooldownError(HotWaterGasAIError):
    """The same Steam search term was queried too recently.

    HTTP 429 - the caller should wait before retrying. The cooldown is
    a per-process, in-memory guard; it is not shared between workers
    and it does not survive a restart.
    """

    http_status = 429
    public_code = "steam_search_cooldown"

    def __init__(self, message: str = "", *, outcome: str = "cooldown") -> None:
        super().__init__(message or self.__class__.__name__)
        self.outcome = outcome


class SteamAppIdInvalidError(HotWaterGasAIError):
    """The supplied Steam App ID is not a usable identifier.

    This is a caller error (HTTP 422). The route layer returns a canned
    message; the diagnostic detail stays in the logs.
    """

    http_status = 422
    public_code = "invalid_steam_app_id"

    def __init__(self, message: str = "", *, outcome: str = "invalid_app_id") -> None:
        super().__init__(message or self.__class__.__name__)
        self.outcome = outcome
