"""English→Vietnamese description translation endpoint.

This router owns the ``POST /internal/descriptions/generate`` route:
same path, same request schema, same response schema, same status codes
as before.

The endpoint lives on its own path
(``POST /internal/descriptions/generate``) and returns the
translation contract:

* the payload is a *translation*, never a verified fact set,
* the translated description is **always** ``unverified`` - the service
  translates, it does not check,
* the release date, the platform availability and the **Windows-only**
  system requirements are deterministic, and they are reported
  separately from the prose,
* the ``warnings`` collection carries stable codes and severities, so a
  consumer can tell a truncated translation from a short one without
  parsing prose,
* the ``description_source`` block describes what was translated.

The route never logs request bodies, tokens, or generated content.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.dependencies.auth import require_internal_token
from app.errors import (
    GenerationDisabledError,
    MissingOllamaConfigError,
)
from app.errors_http import (
    _error_payload,
    _error_response,
    register_domain_error_handler,
    register_missing_ollama_handler,
)
from app.schemas.description_fields import ErrorResponse
from app.schemas.description import (
    GenerateDescriptionRequest,
    GenerateDescriptionResponse,
)
from app.services.generation import DescriptionTranslationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal/descriptions", tags=["descriptions"])


# ── Helpers ───────────────────────────────────────────────────────────────


def _service(request: Request) -> DescriptionTranslationService:
    """Return the application-scoped translation service.

    The lifespan handler attaches the service to ``app.state``. A
    missing service is a 500 so the misconfiguration is visible.
    """

    service = getattr(request.app.state, "translation_service", None)
    if service is None:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500,
            detail=_error_payload(
                "service_not_initialised",
                "service_not_initialised",
            ),
        )
    return service


# ── Exception handlers ────────────────────────────────────────────────────


def register_handlers_for_descriptions(app: Any) -> None:
    """Register the domain-exception handlers used by the route.

    The application-wide bundle already registers a handler for
    :class:`app.errors.HotWaterGasAIError`; FastAPI keeps the most
    recently registered handler for a given exception class, so
    registering it again here would be redundant. This function exists
    so the router can be mounted on a standalone app in tests
    without depending on the application factory's registration.
    """

    register_domain_error_handler(app)
    register_missing_ollama_handler(app)


# ── Routes ────────────────────────────────────────────────────────────────


#: The canonical path of the translation route.
#:
#: The route used to live under a version-suffixed path. The contract
#: itself was never versioned — the version travelled *inside* the
#: payload, in ``schema_version`` — so the path carried a version it
#: did not need.
CANONICAL_PATH = "/generate"

#: The path the route was published under before the rename.
#:
#: It is kept as a temporary alias so a deployment that has not yet
#: been updated keeps working. The alias is deliberately **excluded**
#: from the OpenAPI document: the published contract must show exactly
#: one route, or a consumer cannot tell which one is canonical.
LEGACY_PATH = "/generate-v3"


@router.post(
    CANONICAL_PATH,
    response_model=GenerateDescriptionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": (
                "Translation generated successfully. The 'warnings' "
                "collection is always present and carries stable codes: "
                "'source_truncated' (the Steam description was longer than "
                "the source budget and only the beginning was translated), "
                "'source_block_dropped' (individual blocks did not fit), "
                "'source_insufficient' (the selected English source carried "
                "less material than the target range needs - a statement "
                "about the source, not about the translation), "
                "'translation_below_target' (the translated text is shorter "
                "than the soft target range - a statement about the "
                "translation, not about the source) and "
                "'description_over_ceiling' (the translation exceeded the "
                "hard ceiling and the call failed). The translated "
                "description is always 'unverified': the service "
                "translates, it does not check."
            )
        },
        401: {"model": ErrorResponse, "description": "Missing or invalid internal token."},
        404: {"model": ErrorResponse, "description": "The Steam App ID does not identify a retrievable app."},
        409: {"model": ErrorResponse, "description": "The Steam App ID identifies a different game than the supplied product name."},
        413: {"model": ErrorResponse, "description": "The request or the assembled provider input exceeds the configured size limit."},
        422: {"model": ErrorResponse, "description": "Request payload is invalid."},
        500: {"model": ErrorResponse, "description": "Server-side misconfiguration."},
        502: {"model": ErrorResponse, "description": "Upstream provider is unavailable or rejected the request."},
        503: {"model": ErrorResponse, "description": "AI generation is disabled or not fully configured."},
        504: {"model": ErrorResponse, "description": "Upstream provider timed out."},
    },
    dependencies=[Depends(require_internal_token)],
)
def generate_description(
    payload: GenerateDescriptionRequest,
    request: Request,
) -> GenerateDescriptionResponse:
    """Translate the English Steam description into Vietnamese.

    Authentication is performed by the ``require_internal_token``
    dependency declared on the route, so the handler may assume the
    request is authenticated.

    The Steam retrieval is **required** for this path: without an
    App ID there is no English source to translate. A Steam failure is
    reported as an HTTP error rather than as a translation of nothing.
    """

    settings = getattr(request.app.state, "settings", None)
    if settings is None:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500,
            detail=_error_payload(
                "service_not_initialised",
                "service_not_initialised",
            ),
        )

    if not settings.ai_generation_enabled:
        raise GenerationDisabledError()
    if settings.llm_provider != "ollama":
        # The contract currently has a single implementation. The
        # check is explicit so the operator gets a controlled 503
        # instead of a confusing schema error.
        raise MissingOllamaConfigError()

    service = _service(request)
    return service.generate(payload)


@router.post(
    LEGACY_PATH,
    response_model=GenerateDescriptionResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
    dependencies=[Depends(require_internal_token)],
)
def generate_description_legacy(
    payload: GenerateDescriptionRequest,
    request: Request,
) -> GenerateDescriptionResponse:
    """Deprecated alias for :func:`generate_description`.

    The path is the one the route was published under before the
    rename. It is a thin delegation: the same handler serves both
    paths, so the request schema, the response schema and the status
    codes are identical.

    It is excluded from the OpenAPI document, so the published contract
    advertises only the canonical path.
    """

    return generate_description(payload, request)


__all__ = [
    "router",
    "register_handlers_for_descriptions",
    "CANONICAL_PATH",
    "LEGACY_PATH",
    "GenerateDescriptionRequest",
    "GenerateDescriptionResponse",
]
