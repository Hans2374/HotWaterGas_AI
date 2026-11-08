"""Shared HTTP error envelope and exception-handler registration.

This module is **version-neutral**. It holds the pieces every router
needs to turn a domain error into a stable HTTP response:

* the canonical error envelope (``{"code": ..., "message": ...}``),
* the canned public message for every documented status code,
* the mapping from a domain exception to a sanitised response, and
* the registration of the exception handlers.

Why the module exists
---------------------
The helpers used to live inside the first description router.
The description router and the Steam router both imported them from
there, so the routers that had to be retained depended on a router
that was scheduled for removal. The helpers themselves had nothing
to do with that endpoint, so they belong in a module of their own.
That router has now been removed, and every retained router imports
these helpers from here.

Safety of the error surface
---------------------------
The route layer **never** echoes an exception's own message to the
client: the underlying provider or SDK may have embedded credentials,
file paths or other sensitive content in it. Only the canned message
below is ever returned, and the exception *class* is the only thing
that decides the response. The public code and the HTTP status are
carried by the exception class itself
(:mod:`app.errors`).

The canned messages are the single source of truth for the wording of
every documented error, so a client that branches on ``code`` never has
to parse prose.

Registration
------------
Every handler is defined here exactly once. The application-wide bundle
is installed by :func:`register_handlers`, which the application factory
calls a single time. The per-router helpers
(:func:`register_domain_error_handler` and
:func:`register_missing_ollama_handler`) exist so a router can still be
mounted on a standalone app — in a test, say — without depending on the
full bundle.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.errors import (
    HotWaterGasAIError,
    InvalidInternalTokenError,
    MissingInternalTokenConfigError,
    MissingOllamaConfigError,
    RequestTooLargeError,
)


# Safe, generic messages for every documented status code. The route
# layer NEVER echoes the original exception's message to the client
# because the underlying provider / SDK may embed credentials, file
# paths, or other sensitive content in the message. The mapping below
# is the single source of truth.
_GENERIC_MESSAGES: dict[str, str] = {
    "generation_disabled": "AI generation is currently disabled on this server.",
    "internal_token_not_configured": "The server is not configured to accept internal requests.",
    "unauthorized": "Missing or invalid internal token.",
    "provider_not_configured": "The server is not configured to accept generation requests.",
    "provider_auth_failed": "Upstream provider rejected the request.",
    "provider_model_not_found": "Configured provider model is not available.",
    "rate_limited": "Upstream provider is currently rate-limiting our requests.",
    "provider_timeout": "Upstream provider timed out before responding.",
    "provider_unavailable": "Upstream provider is currently unavailable.",
    "invalid_provider_output": "Upstream provider returned unusable output.",
    "request_too_large": "Request payload exceeds the configured size limit.",
    "insufficient_product_data": (
        "At least one of developer, publisher, platform, release_date, "
        "categories, tags, short_facts, or system_requirements must be "
        "supplied alongside product_name."
    ),
    "invalid_request": "The request payload is invalid.",
    "service_not_initialised": "The generation service is not initialised.",
    "steam_unavailable": "Steam metadata could not be retrieved.",
    "steam_not_found": "The Steam App ID does not identify a retrievable app.",
    "steam_identity_mismatch": (
        "The Steam App ID refers to a different game than the supplied "
        "product name."
    ),
    "invalid_steam_app_id": "The Steam App ID is not a valid identifier.",
    "invalid_steam_search_query": "The Steam search query is not usable.",
    "steam_search_cooldown": "This Steam search was made too recently.",
    "steam_search_disabled": "Steam retrieval is currently disabled on this server.",
    "steam_rate_limited": "Steam is currently rate-limiting our requests.",
    "steam_search_unavailable": "Steam search could not be completed.",
}


def _error_payload(code: str, message: str) -> Dict[str, str]:
    """Build the canonical error envelope."""

    return {"code": code, "message": message}


def _safe_message(code: str) -> str:
    """Return the canned public message for a public code.

    Falls back to a generic string if the code is unknown so the route
    layer never echoes an arbitrary exception message.
    """

    return _GENERIC_MESSAGES.get(
        code, "An unexpected error occurred while processing the request."
    )


def _error_response(exc: HotWaterGasAIError) -> JSONResponse:
    """Map a domain exception to a sanitised JSON response.

    The original exception's ``message`` is intentionally **not**
    included in the response body; only the canned public message is
    returned. The exception class is the only thing we trust.
    """

    return JSONResponse(
        status_code=exc.http_status,
        content=_error_payload(exc.public_code, _safe_message(exc.public_code)),
    )


# ── Individual exception handlers ─────────────────────────────────────────


def register_domain_error_handler(app: Any) -> None:
    """Register the handler for the base domain error.

    FastAPI keeps the most recently registered handler for a given
    exception class, so this handler also covers every subclass of
    :class:`app.errors.HotWaterGasAIError` that has no handler of its
    own.
    """

    @app.exception_handler(HotWaterGasAIError)
    async def _domain_error_handler(_: Request, exc: HotWaterGasAIError) -> JSONResponse:  # noqa: D401
        return _error_response(exc)


def register_missing_ollama_handler(app: Any) -> None:
    """Register the handler for the missing-Ollama-configuration error.

    The error is a subclass of the base domain error and maps onto the
    same "provider is not configured" contract, but it is registered
    explicitly so the behaviour does not depend on the MRO lookup.
    """

    @app.exception_handler(MissingOllamaConfigError)
    async def _missing_ollama_handler(
        _: Request, exc: MissingOllamaConfigError
    ) -> JSONResponse:  # noqa: D401
        return _error_response(exc)


def register_validation_handler(app: Any) -> None:
    """Register the handler for a request-payload validation failure."""

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:  # noqa: D401
        # Pydantic validation errors include the offending field path,
        # which is useful for the caller and contains no secrets. We
        # still wrap the response in our envelope for consistency.
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_payload(
                "invalid_request",
                "The request payload is invalid.",
            ),
            headers={"X-Validation-Errors": "1"},
        )


def register_http_exception_handler(app: Any) -> None:
    """Register the handler that re-shapes an ``HTTPException``."""

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(
        _: Request, exc: HTTPException
    ) -> JSONResponse:  # noqa: D401
        # The route layer raises ``HTTPException(422, detail={...})``
        # for the "only product_name" case. We re-shape it into the
        # canonical envelope and never echo the original ``detail``
        # (which may include arbitrary strings) without filtering.
        code: str
        message: str
        if isinstance(exc.detail, dict) and "code" in exc.detail and "message" in exc.detail:
            code = str(exc.detail["code"])
            # Use the canned message instead of the supplied one.
            message = _safe_message(code) if code in _GENERIC_MESSAGES else "The request payload is invalid."
        else:
            code = "invalid_request"
            message = "The request payload is invalid."
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(code, message),
        )


def register_request_too_large_handler(app: Any) -> None:
    """Register the handler for an oversized provider input."""

    @app.exception_handler(RequestTooLargeError)
    async def _too_large_handler(_: Request, exc: RequestTooLargeError) -> JSONResponse:  # noqa: D401
        return _error_response(exc)


# ── Handler bundles ───────────────────────────────────────────────────────


def register_handlers(app: Any) -> None:
    """Register the complete, application-wide exception-handler bundle.

    Called from ``app/main.py`` during app construction. The handlers
    live here rather than in a router so that the whole error surface
    is defined in one place, and so that the retained routers do not
    depend on a router that is scheduled for removal.

    Each exception class is registered **exactly once**: the bundle is
    the single registration point for the application.
    """

    register_domain_error_handler(app)
    register_missing_ollama_handler(app)
    register_validation_handler(app)
    register_http_exception_handler(app)
    register_request_too_large_handler(app)


def install_auth_handlers(app: Any) -> None:
    """Register only the auth-related exception handlers on ``app``.

    Useful for tests that exercise a minimal app with only the
    ``require_internal_token`` dependency mounted. The full handler
    bundle is more than the minimal apps need, so this lighter entry
    point keeps the auth tests focused on the dependency itself.
    """

    @app.exception_handler(MissingInternalTokenConfigError)
    async def _missing_cfg_handler(
        _: Request, exc: MissingInternalTokenConfigError
    ) -> JSONResponse:  # noqa: D401
        return JSONResponse(
            status_code=exc.http_status,
            content=_error_payload(exc.public_code, _safe_message(exc.public_code)),
        )

    @app.exception_handler(InvalidInternalTokenError)
    async def _invalid_token_handler(
        _: Request, exc: InvalidInternalTokenError
    ) -> JSONResponse:  # noqa: D401
        return JSONResponse(
            status_code=exc.http_status,
            content=_error_payload(exc.public_code, _safe_message(exc.public_code)),
        )


__all__ = [
    "register_handlers",
    "install_auth_handlers",
    "register_domain_error_handler",
    "register_missing_ollama_handler",
    "register_validation_handler",
    "register_http_exception_handler",
    "register_request_too_large_handler",
]
