"""Optional enrichment of the generated description from public Steam metadata.

The retrieval layer is **additive** and **optional**. The generation
route uses it only when the operator opts in.

The layer is split into five pieces so that each one stays testable
without a network:

``app.retrieval.steam``
    The HTTP boundary. One fixed endpoint, strict timeouts, a bounded
    body, a content-type check and no automatic retry.

``app.retrieval.steam_mapper``
    The pure mapping from a Steam payload to facts and to
    service-owned evidence records.

``app.retrieval.description_source``
    The selection and parsing of the English store description that the
    translation path uses: headings, paragraphs and lists are preserved,
    repeated blocks are removed conservatively, and the source is cut
    at whole-block boundaries when it exceeds the source budget.

``app.retrieval.steam_search``
    The HTTP boundary of the Steam Store Search endpoint, which is
    independent of the evidence pipeline.

``app.retrieval.steam_search_mapper``
    The pure mapping from a Steam Store Search payload to search
    results.
"""

from __future__ import annotations

from app.retrieval.steam import (
    MAX_STEAM_APP_ID,
    SteamAppIdRetriever,
    build_appdetails_url,
    validate_app_id,
)
from app.retrieval.steam_mapper import (
    STEAM_APPDETAILS_ENDPOINT,
    STEAM_COUNTRY,
    STEAM_LOCALE,
    map_appdetails,
    parse_steam_date,
    strip_html,
    titles_match,
)
from app.retrieval.steam_models import (
    IdentityCheck,
    SteamFacts,
    SteamRequirements,
    SteamRetrievalResult,
)
from app.retrieval.steam_search import (
    SteamSearchRetriever,
    build_storesearch_url,
    validate_search_term,
)
from app.retrieval.steam_search_mapper import (
    MAX_STEAM_SEARCH_CANDIDATES,
    STEAM_STORESEARCH_ENDPOINT,
    map_storesearch,
)
from app.retrieval.steam_search_models import (
    SteamSearchCandidate,
    SteamSearchPrice,
    SteamSearchResult,
)


__all__ = [
    "IdentityCheck",
    "MAX_STEAM_APP_ID",
    "MAX_STEAM_SEARCH_CANDIDATES",
    "STEAM_APPDETAILS_ENDPOINT",
    "STEAM_COUNTRY",
    "STEAM_LOCALE",
    "STEAM_STORESEARCH_ENDPOINT",
    "SteamAppIdRetriever",
    "SteamFacts",
    "SteamRequirements",
    "SteamRetrievalResult",
    "SteamSearchCandidate",
    "SteamSearchPrice",
    "SteamSearchResult",
    "SteamSearchRetriever",
    "build_appdetails_url",
    "build_storesearch_url",
    "map_appdetails",
    "map_storesearch",
    "parse_steam_date",
    "strip_html",
    "titles_match",
    "validate_app_id",
    "validate_search_term",
]
