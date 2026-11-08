"""Internal Steam Store Search endpoint.

This router owns the ``POST /internal/steam/search`` route: same path,
same request schema, same response schema, same status codes as before.
The generation routers are untouched.

The route exposes the Steam **search** core — a lookup that answers
"which Steam apps might this game be?" — behind the same internal-token
dependency as the generation endpoints.

What the endpoint does *not* do
--------------------------------
It does not enrich, verify or complete anything. A search result is a
**candidate**, not a fact:

* it never creates evidence, and it never touches the evidence
  pipeline,
* it never calls ``appdetails`` for a candidate, so a search costs
  exactly one upstream request,
* it never filters by product type. Steam reports ``type="app"`` for a
  full game, a demo and a soundtrack alike, so filtering belongs to the
  later ``appdetails`` stage.

The route never logs the query, the token or the upstream response body.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.dependencies.auth import require_internal_token
from app.errors import (
    SteamRetrievalError,
    SteamSearchDisabledError,
    SteamSearchRateLimitError,
    SteamSearchUnavailableError,
)
from app.errors_http import (
    _error_payload,
    register_domain_error_handler,
)
from app.schemas.description_fields import ErrorResponse
from app.schemas.steam_search import (
    SteamSearchCandidateResponse,
    SteamSearchRequest,
    SteamSearchResponse,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal/steam", tags=["steam"])


# ── Helpers ───────────────────────────────────────────────────────────────


def _settings(request: Request) -> Any:
    """Return the application-scoped settings."""

    settings = getattr(request.app.state, "settings", None)
    if settings is None:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500,
            detail=_error_payload(
                "service_not_initialised",
                "service_not_initialised",
            ),
        )
    return settings


def _retriever(request: Request) -> Any:
    """Return the application-scoped Steam search retriever.

    The lifespan handler attaches the retriever to ``app.state`` only
    when the operator has enabled Steam retrieval. A missing retriever
    is a 500 so the misconfiguration is visible.
    """

    retriever = getattr(request.app.state, "steam_search_retriever", None)
    if retriever is None:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500,
            detail=_error_payload(
                "service_not_initialised",
                "service_not_initialised",
            ),
        )
    return retriever


def _translate_upstream_error(exc: SteamRetrievalError) -> HotWaterGasAIError:
    """Map an upstream retrieval failure onto a search-specific error.

    The core reports every upstream failure as
    :class:`SteamRetrievalError` with an ``outcome`` code. The route
    turns the two outcomes that deserve their own status into the
    matching search error, and folds everything else into a single
    "upstream unavailable" error.

    The original exception is never propagated to the client: it may
    embed the upstream body or the request URL.
    """

    if exc.outcome == "rate_limited":
        # Steam itself is throttling us. The caller should back off.
        return SteamSearchRateLimitError(outcome="rate_limited")
    return SteamSearchUnavailableError(outcome=exc.outcome)


# ── Exception handlers ────────────────────────────────────────────────────


def register_steam_search_handlers(app: Any) -> None:
    """Register the domain-exception handlers used by the search route.

    The application-wide bundle already registers a handler for
    :class:`app.errors.HotWaterGasAIError`; FastAPI keeps the most
    recently registered handler for a given exception class, so
    registering it again here would be redundant. This function exists
    so the search router can be mounted on a standalone app in tests
    without depending on the application factory's registration.
    """

    register_domain_error_handler(app)


# ── Routes ────────────────────────────────────────────────────────────────


@router.post(
    "/search",
    response_model=SteamSearchResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": (
                "Search completed successfully. 'candidates' holds the "
                "usable candidates in Steam's relevance order; it is "
                "empty when Steam reported no usable match. 'total' "
                "counts Steam's own matches and may exceed the number "
                "of returned candidates."
            )
        },
        401: {"model": ErrorResponse, "description": "Missing or invalid internal token."},
        422: {"model": ErrorResponse, "description": "The search query is missing, empty or too long."},
        429: {
            "model": ErrorResponse,
            "description": (
                "The same query was searched too recently, or Steam is "
                "rate-limiting the request."
            ),
        },
        500: {"model": ErrorResponse, "description": "Server-side misconfiguration."},
        502: {"model": ErrorResponse, "description": "Steam could not be reached or returned an unusable response."},
        503: {"model": ErrorResponse, "description": "Steam retrieval is disabled on this server."},
    },
    dependencies=[Depends(require_internal_token)],
)
def search_steam_store(
    payload: SteamSearchRequest,
    request: Request,
) -> SteamSearchResponse:
    """Search the Steam Store for candidates matching the query.

    Authentication is performed by the ``require_internal_token``
    dependency declared on the route, so the handler may assume the
    request is authenticated.

    The feature switch is checked first: when
    ``steam_retrieval_enabled`` is off, no upstream request is made and
    the call fails with HTTP 503. The switch is not enabled by default.

    The query itself is validated by the search core, which trims it
    and rejects an empty or over-long value with HTTP 422.
    """

    settings = _settings(request)

    if not settings.steam_retrieval_enabled:
        logger.warning("steam_search_disabled")
        raise SteamSearchDisabledError(
            "Steam retrieval is disabled on this server.",
            outcome="disabled",
        )

    retriever = _retriever(request)

    try:
        result = retriever.search(payload.query)
    except SteamRetrievalError as exc:
        # The core reports every upstream failure this way. The route
        # maps it onto a stable, safe status code. The upstream body is
        # never included in the response.
        logger.warning("steam_search_upstream_failed outcome=%s", exc.outcome)
        raise _translate_upstream_error(exc) from None

    candidates = [
        SteamSearchCandidateResponse(
            app_id=candidate.app_id,
            name=candidate.name,
            type=candidate.app_type,
            tiny_image=candidate.tiny_image,
        )
        for candidate in result.candidates
    ]

    return SteamSearchResponse(
        total=result.total,
        candidate_count=len(candidates),
        invalid_count=result.invalid_count,
        truncated_count=result.truncated_count,
        candidates=candidates,
    )


__all__ = [
    "router",
    "register_steam_search_handlers",
    "SteamSearchCandidateResponse",
    "SteamSearchRequest",
    "SteamSearchResponse",
]
