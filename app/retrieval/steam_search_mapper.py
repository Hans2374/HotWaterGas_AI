"""Pure parsing of a Steam Store Search payload into search candidates.

Everything in this module is a **pure function** of its input. There is
no network access, no clock access and no I/O, so every mapping rule can
be tested offline against the captured fixture.

The observed contract
---------------------
A real Store Search response (``tests/fixtures/steam/storesearch_portal_2.json``)
looks like this::

    {"total": 10, "items": [ {...}, {...} ]}

``total`` is an integer and ``items`` is an array. Each item carries:

==================  ========  ==========================================
Field               Required  Notes
==================  ========  ==========================================
``id``              yes       Integer app id. Must be a positive int.
``name``            yes       Title. Must be a non-empty string.
``type``            no        Observed: ``"app"``. Informational only.
``tiny_image``      no        Absolute https URL of the capsule image.
``price``           no        ``{currency, initial, final}``.
``metascore``       no        String; ``""`` when there is no score.
``platforms``        no        ``{windows, mac, linux}`` booleans.
``streamingvideo``  no        Boolean.
``controller_support`` no     String, e.g. ``"full"``.
==================  ========  ==========================================

Almost every optional field is missing from *some* item in the captured
fixture, and ``price`` is missing from six of its ten items. Optional
fields are therefore always optional here.

What this module deliberately does **not** do
---------------------------------------------
It does **not** filter by product type. Steam's Store Search API
reports ``type="app"`` for a full game, a soundtrack and a demo alike -
the captured fixture contains all three. Deciding what a candidate
really is belongs to the later ``appdetails`` stage, not here.

It does **not** create evidence. A candidate is a pointer, not a fact.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from app.retrieval.steam_search_models import (
    SteamSearchCandidate,
    SteamSearchPrice,
    SteamSearchResult,
)


#: The single allow-listed Steam Store Search endpoint. There is no other.
STEAM_STORESEARCH_ENDPOINT = "https://store.steampowered.com/api/storesearch/"

#: The fixed locale used for every request. Steam localises some fields
#: (notably ``price.currency`` and item titles), so the locale is pinned
#: rather than negotiated.
STEAM_STORESEARCH_LOCALE = "vietnamese"

#: The fixed country used for every request, for the same reason.
STEAM_STORESEARCH_COUNTRY = "vn"

#: Maximum number of candidates returned to the caller. Steam may report
#: more items than this; the surplus is discarded rather than mapped.
MAX_STEAM_SEARCH_CANDIDATES = 15

#: The Steam platform keys mapped to canonical names, in a fixed order so
#: the resulting list is deterministic.
_PLATFORM_KEYS: Tuple[Tuple[str, str], ...] = (
    ("windows", "Windows"),
    ("mac", "macOS"),
    ("linux", "Linux"),
)

#: Maximum length of a candidate title. Steam titles are short; a
#: pathological one is truncated rather than rejected, because the App ID
#: is what makes a candidate useful.
MAX_CANDIDATE_NAME_CHARS = 200

#: Maximum length of a candidate image URL.
MAX_CANDIDATE_IMAGE_CHARS = 512


class SteamSearchPayloadError(ValueError):
    """The Store Search payload does not have the expected top-level shape.

    Raised for a payload that is not an object, that lacks the ``items``
    array, or whose ``items`` is not an array. The caller translates
    this into a domain error; it never carries the response body.
    """


def map_storesearch(
    payload: Any,
    *,
    max_candidates: int = MAX_STEAM_SEARCH_CANDIDATES,
) -> SteamSearchResult:
    """Map a Store Search payload into a :class:`SteamSearchResult`.

    Items that do not carry a usable App ID or title are **skipped**,
    not fatal: a single malformed entry must not lose the whole result
    set. The number of skipped entries is reported as
    ``invalid_count`` and the number dropped by the candidate cap as
    ``truncated_count``.

    Duplicate App IDs are collapsed - the first occurrence wins, because
    Steam's ordering is its own relevance ranking and the first
    occurrence is the most relevant one.
    """

    if max_candidates < 0:
        raise ValueError("max_candidates must not be negative")

    if not isinstance(payload, dict):
        raise SteamSearchPayloadError("search payload must be an object")

    items = payload.get("items")
    if not isinstance(items, list):
        raise SteamSearchPayloadError("search payload has no items array")

    total = _map_total(payload.get("total"), len(items))

    candidates: List[SteamSearchCandidate] = []
    invalid_count = 0
    truncated_count = 0
    seen_ids = set()

    for item in items:
        if len(candidates) >= max_candidates:
            # The cap is reached: the remaining items are not even
            # validated, they are simply dropped.
            truncated_count += 1
            continue

        candidate = _map_item(item)
        if candidate is None:
            invalid_count += 1
            continue

        if candidate.app_id in seen_ids:
            continue

        seen_ids.add(candidate.app_id)
        candidates.append(candidate)

    return SteamSearchResult(
        total=total,
        candidates=candidates,
        invalid_count=invalid_count,
        truncated_count=truncated_count,
    )


# ── Helpers ───────────────────────────────────────────────────────────────


def _map_total(value: Any, item_count: int) -> int:
    """Return the reported total, falling back to the item count.

    ``total`` is documented as an integer but Steam is not contractually
    bound by it here, so a missing or nonsensical value falls back to the
    number of items actually returned rather than failing the request.
    """

    if isinstance(value, bool):
        return item_count
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return item_count


def _map_item(item: Any) -> Optional[SteamSearchCandidate]:
    """Map one raw item, or return ``None`` when it is not usable."""

    if not isinstance(item, dict):
        return None

    app_id = _map_app_id(item.get("id"))
    if app_id is None:
        return None

    name = _map_name(item.get("name"))
    if name is None:
        return None

    return SteamSearchCandidate(
        app_id=app_id,
        name=name,
        app_type=_map_optional_str(item.get("type")),
        tiny_image=_map_image(item.get("tiny_image")),
        price=_map_price(item.get("price")),
        metascore=_map_metascore(item.get("metascore")),
        platforms=_map_platforms(item.get("platforms")),
        streamingvideo=_map_optional_bool(item.get("streamingvideo")),
        controller_support=_map_optional_str(item.get("controller_support")),
    )


def _map_app_id(value: Any) -> Optional[int]:
    """Return the App ID when it is a positive integer, else ``None``.

    ``bool`` is rejected explicitly: it is an ``int`` subclass in
    Python but is never a valid App ID. A numeric string is accepted
    because the identifier is what makes a candidate usable.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text.isdigit():
            return None
        value = int(text)
    if not isinstance(value, int):
        return None
    if value < 1:
        return None
    return value


def _map_name(value: Any) -> Optional[str]:
    """Return the title when it is a non-empty string, else ``None``."""

    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped[:MAX_CANDIDATE_NAME_CHARS]


def _map_optional_str(value: Any) -> Optional[str]:
    """Return a trimmed non-empty string, else ``None``."""

    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _map_image(value: Any) -> Optional[str]:
    """Return the capsule image URL when it is an http(s) URL.

    A non-URL or a non-http scheme is dropped: the value is only ever
    used for display, and an unusable value must not reach a client.
    """

    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if not stripped.startswith(("https://", "http://")):
        return None
    return stripped[:MAX_CANDIDATE_IMAGE_CHARS]


def _map_price(value: Any) -> Optional[SteamSearchPrice]:
    """Map the optional price block."""

    if not isinstance(value, dict):
        return None
    return SteamSearchPrice(
        currency=_map_optional_str(value.get("currency")),
        initial=_map_number(value.get("initial")),
        final=_map_number(value.get("final")),
    )


def _map_number(value: Any) -> Optional[float]:
    """Return a numeric value, rejecting booleans and non-numbers."""

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _map_metascore(value: Any) -> Optional[int]:
    """Map the metascore.

    Steam reports the score as a string and uses ``""`` for "no score".
    Both the empty string and a non-numeric value map to ``None``.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text.isdigit():
            return None
        return int(text)
    return None


def _map_platforms(value: Any) -> List[str]:
    """Map the platform flags into a deterministic list of names."""

    if not isinstance(value, dict):
        return []
    names = []
    for key, label in _PLATFORM_KEYS:
        if value.get(key) is True:
            names.append(label)
    return names


def _map_optional_bool(value: Any) -> Optional[bool]:
    """Return the boolean when it really is a boolean."""

    if isinstance(value, bool):
        return value
    return None


__all__ = [
    "MAX_CANDIDATE_IMAGE_CHARS",
    "MAX_CANDIDATE_NAME_CHARS",
    "MAX_STEAM_SEARCH_CANDIDATES",
    "STEAM_STORESEARCH_COUNTRY",
    "STEAM_STORESEARCH_ENDPOINT",
    "STEAM_STORESEARCH_LOCALE",
    "SteamSearchPayloadError",
    "map_storesearch",
]
