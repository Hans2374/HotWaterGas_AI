"""Request/response schemas for the internal Steam Store Search endpoint.

The schemas are **additive**: the description schemas in
:mod:`app.schemas.description` are untouched and keep their exact
behaviour.

What the response exposes
-------------------------
A Steam Store Search result is a *candidate*, not a fact. The endpoint
therefore exposes only what a caller needs to **pick** a candidate:

* the App ID (the identifier the ``appdetails`` endpoint accepts),
* the title,
* the capsule image URL (optional),
* the raw ``type`` string (optional, informational only).

Fields that are not needed to make that choice - the price block, the
metascore, the platform flags, the streaming-video flag - are **not**
exposed. They are product facts, and a search result is not a source of
product facts. Exposing them here would invite a caller to treat a
search hit as verified metadata.

The endpoint never filters by product type: Steam reports ``type="app"``
for a full game, a demo and a soundtrack alike, so the filter belongs to
the later ``appdetails`` stage.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class SteamSearchRequest(BaseModel):
    """Payload accepted by ``POST /internal/steam/search``.

    Only the query is accepted. The endpoint has no URL, host, App ID or
    pagination parameters: the Steam endpoint is fixed and the candidate
    count is capped server-side, so a caller cannot widen the upstream
    request.

    The query is validated by the search core, which trims it, rejects
    an empty value and rejects an over-long one. The schema therefore
    only enforces that a string is present; the authoritative check -
    and the one that produces the documented 422 - lives in the core.
    """

    query: str = Field(
        ...,
        description="The game title to search the Steam Store for.",
    )


class SteamSearchCandidateResponse(BaseModel):
    """One candidate returned by the search endpoint.

    Only the fields needed to choose a candidate are present. ``type``
    is Steam's own raw string and is informational only: it does not
    identify the product type of the app.
    """

    model_config = ConfigDict(extra="forbid")

    app_id: int = Field(
        ...,
        gt=0,
        description="The Steam App ID of the candidate.",
    )
    name: str = Field(
        ...,
        min_length=1,
        description="The candidate's title as Steam reports it.",
    )
    type: Optional[str] = Field(
        default=None,
        max_length=64,
        description=(
            "Steam's raw type string (observed: 'app'). Informational "
            "only - it does not distinguish a full game from a demo or "
            "a soundtrack."
        ),
    )
    tiny_image: Optional[str] = Field(
        default=None,
        max_length=512,
        description="URL of the candidate's capsule image, when Steam reports one.",
    )


class SteamSearchResponse(BaseModel):
    """Stable response contract for the Steam search endpoint.

    The response carries the candidates plus the two counters the caller
    needs to interpret them: how many results Steam reported in total,
    and how many returned items were discarded as unusable. The number
    of candidates actually returned is always present.
    """

    model_config = ConfigDict(extra="forbid")

    total: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "The number of results Steam reports for the query, when "
            "available. It counts Steam's matches, not the candidates "
            "returned: Steam may report more matches than the "
            "candidate cap allows."
        ),
    )
    candidate_count: int = Field(
        ...,
        ge=0,
        description="The number of candidates returned in this response.",
    )
    invalid_count: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "The number of returned items that were discarded because "
            "they carried no usable App ID or title, when available."
        ),
    )
    truncated_count: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "The number of valid candidates dropped because the "
            "candidate cap had already been reached, when available."
        ),
    )
    candidates: List[SteamSearchCandidateResponse] = Field(
        default_factory=list,
        description=(
            "The candidates, in Steam's relevance order. Empty when "
            "Steam reported no usable match."
        ),
    )


__all__ = [
    "SteamSearchCandidateResponse",
    "SteamSearchRequest",
    "SteamSearchResponse",
]
