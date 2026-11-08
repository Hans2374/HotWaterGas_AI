"""Data structures produced by the Steam **search** layer.

The search layer is deliberately separate from the ``appdetails``
retrieval layer. ``appdetails`` produces *facts and evidence* for a game
whose App ID is already known; the search layer produces *candidates* -
thin pointers to apps that might be the game an admin is looking for.

A candidate is **not** a fact. It carries an App ID, a title and a couple
of optional display fields. Nothing in this module may be turned into
evidence, and no candidate implies anything about the product type of
the app it points at: the Steam Store Search API reports ``type="app"``
for a full game, a demo and a soundtrack alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class SteamSearchPrice:
    """The price block of a search result.

    Steam reports the price in the currency of the requested country
    (here always VND, because the country parameter is pinned). All
    three fields are optional: the price block is absent for some
    results entirely.
    """

    currency: Optional[str] = None
    initial: Optional[float] = None
    final: Optional[float] = None


@dataclass(frozen=True)
class SteamSearchCandidate:
    """One Steam app returned by the Store Search endpoint.

    Only ``app_id`` and ``name`` are guaranteed to be present - they are
    the two fields the search mapper requires. Everything else is
    optional and stays ``None``/empty when Steam did not report it.

    ``app_id`` is the value Steam reports as ``id``. It is treated as
    an *app* id because that is the identifier the ``appdetails``
    endpoint accepts, but the search endpoint does not distinguish apps
    from other store entries, so the value must not be trusted to
    describe a full game.
    """

    app_id: int
    name: str
    #: Steam's own ``type`` field (observed: ``"app"``). Informational
    #: only - it does not identify the product type.
    app_type: Optional[str] = None
    tiny_image: Optional[str] = None
    price: Optional[SteamSearchPrice] = None
    #: The metascore as a number. Steam reports it as a string and
    #: uses an empty string when there is no score; both the empty
    #: string and a non-numeric value map to ``None``.
    metascore: Optional[int] = None
    #: The platforms Steam reports as available, in a fixed order.
    platforms: List[str] = field(default_factory=list)
    streamingvideo: Optional[bool] = None
    controller_support: Optional[str] = None


@dataclass(frozen=True)
class SteamSearchResult:
    """The outcome of one successful Store Search request.

    ``total`` is the number of results Steam reports for the query.
    ``candidates`` holds only the *valid* candidates, capped at the
    configured maximum. ``invalid_count`` records how many returned
    items were discarded because they did not carry a usable App ID or
    title - it is a diagnostic, never an error.
    """

    total: int
    candidates: List[SteamSearchCandidate] = field(default_factory=list)
    invalid_count: int = 0
    #: How many valid candidates were dropped because the maximum
    #: number of candidates had already been collected.
    truncated_count: int = 0


__all__ = [
    "SteamSearchCandidate",
    "SteamSearchPrice",
    "SteamSearchResult",
]
