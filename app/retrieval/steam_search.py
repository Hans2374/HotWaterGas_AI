"""The Steam **Store Search** retriever.

This module is the only place in the service that knows the Steam Store
Search HTTP API exists. It returns *candidates* - thin pointers to
apps that might be the game an admin is looking for. It never returns
facts and never produces evidence.

Guardrails
----------
**One fixed endpoint.** The URL is assembled from a module constant and
    the validated search term. There is no way for a caller to supply a
host, a path or a URL, so the service can never be turned into a
general-purpose fetcher.

**Bounded in both time and size.** Separate connect and read timeouts
    keep a stalled connection from consuming the whole request budget, and
    the body is read *while* counting bytes so an oversized response is
    aborted instead of buffered.

**No automatic retry.** A rate-limited or failed request is reported,
    never retried: a silent retry loop would multiply the load on an
    already-struggling upstream and would hide the failure from the caller.

**A repeated-query cooldown.** The same term is not queried twice
    within the cooldown window. The window is enforced **in this
    process only**: it is not distributed between workers and it does
    not survive a restart. A multi-worker deployment therefore enforces
    the window per worker, not globally.

**Nothing sensitive is logged.** Log lines carry the HTTP status, a
    duration, a count and a short outcome code. The search term itself
    is never logged - it is free user text - and the response body is
    never logged or propagated into an error message.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

from app.errors import (
    SteamRetrievalError,
    SteamSearchCooldownError,
    SteamSearchQueryInvalidError,
)
from app.retrieval.steam_search_mapper import (
    MAX_STEAM_SEARCH_CANDIDATES,
    STEAM_STORESEARCH_COUNTRY,
    STEAM_STORESEARCH_ENDPOINT,
    STEAM_STORESEARCH_LOCALE,
    SteamSearchPayloadError,
    map_storesearch,
)
from app.retrieval.steam_search_models import SteamSearchResult


logger = logging.getLogger(__name__)


#: The content types the retriever accepts. Steam answers
#: ``application/json``; anything else is rejected before parsing.
ACCEPTED_CONTENT_TYPES = ("application/json", "text/json")

#: The maximum number of bytes read from the Steam endpoint.
DEFAULT_MAX_RESPONSE_BYTES = 512 * 1024

#: Maximum length of a search term, in characters. A search term is a
#: game title, not a paragraph; the longest legitimate titles are far
#: below this bound.
DEFAULT_MAX_QUERY_LENGTH = 200

#: The repeated-query cooldown, in seconds.
DEFAULT_QUERY_COOLDOWN_SECONDS = 2.0


def build_storesearch_url(term: str) -> str:
    """Return the request URL for one search term.

    The URL is built from the fixed endpoint constant and the already
    validated term. It is the *only* URL this module can produce.
    """

    return (
        f"{STEAM_STORESEARCH_ENDPOINT}"
        f"?term={term}"
        f"&l={STEAM_STORESEARCH_LOCALE}"
        f"&cc={STEAM_STORESEARCH_COUNTRY}"
    )


def validate_search_term(term: Any) -> str:
    """Validate a search term and return its normalised form.

    The term is trimmed and length-checked. The returned value is the
    *encoded* query parameter value, so it can be placed in the URL
    without any further escaping.

    Raises :class:`ValueError` for a term that is not a non-empty
    string within the length bound.
    """

    if not isinstance(term, str):
        raise ValueError("steam search term must be a string")

    stripped = term.strip()
    if not stripped:
        raise ValueError("steam search term must not be empty")
    if len(stripped) > DEFAULT_MAX_QUERY_LENGTH:
        raise ValueError("steam search term is too long")

    return _quote(stripped)


def _quote(value: str) -> str:
    """Percent-encode a query parameter value.

    ``httpx`` would encode the parameter itself, but the URL is built
    by hand here so that the builder stays the single source of truth
    for the request shape. The encoding therefore happens up front.
    """

    return quote(value, safe="")


class SteamSearchRetriever:
    """Search the Steam Store and return candidate apps.

    The retriever performs no retries and never raises a raw transport
    exception: every failure mode is translated into a domain error
    that carries a safe outcome code.
    """

    def __init__(
        self,
        *,
        connect_timeout_seconds: float = 5.0,
        read_timeout_seconds: float = 10.0,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_candidates: int = MAX_STEAM_SEARCH_CANDIDATES,
        cooldown_seconds: float = DEFAULT_QUERY_COOLDOWN_SECONDS,
        client: Optional[httpx.Client] = None,
    ) -> None:
        if connect_timeout_seconds <= 0:
            raise ValueError("connect_timeout_seconds must be positive")
        if read_timeout_seconds <= 0:
            raise ValueError("read_timeout_seconds must be positive")
        if max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")
        if max_candidates < 0:
            raise ValueError("max_candidates must not be negative")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must not be negative")

        self._connect_timeout = float(connect_timeout_seconds)
        self._read_timeout = float(read_timeout_seconds)
        self._max_response_bytes = int(max_response_bytes)
        self._max_candidates = int(max_candidates)
        self._cooldown_seconds = float(cooldown_seconds)

        # A reusable client keeps the connection warm. Tests inject
        # their own transport instead of touching the socket layer.
        self._client = client if client is not None else httpx.Client(
            timeout=httpx.Timeout(
                self._read_timeout, connect=self._connect_timeout
            ),
            follow_redirects=False,
        )
        self._owns_client = client is None

        # The cooldown registry. Process-local by design - see the
        # module docstring. The lock makes the check-then-record
        # sequence atomic so two threads cannot both pass the check.
        self._cooldown_lock = threading.Lock()
        self._last_query_at: Dict[str, float] = {}

    # ── Public API ─────────────────────────────────────────────────────

    def search(self, term: Any) -> SteamSearchResult:
        """Search the Steam Store for ``term``.

        Raises :class:`SteamSearchQueryInvalidError` for an unusable
        term, :class:`SteamSearchCooldownError` when the same term was
        queried too recently, and :class:`SteamRetrievalError` for every
        upstream failure.
        """

        try:
            encoded = validate_search_term(term)
        except ValueError as exc:
            # The term is user-supplied free text: it is never echoed
            # back, neither in the message nor in the log line.
            logger.warning("steam_search_invalid_query")
            raise SteamSearchQueryInvalidError(
                "The Steam search term is not usable.", outcome="invalid_query"
            ) from exc

        self._enforce_cooldown(encoded)

        url = build_storesearch_url(encoded)

        started = time.monotonic()
        payload = self._request_json(url)
        duration_ms = int((time.monotonic() - started) * 1000)

        try:
            result = map_storesearch(payload, max_candidates=self._max_candidates)
        except SteamSearchPayloadError as exc:
            logger.warning("steam_search_unexpected_structure")
            raise SteamRetrievalError(
                "Steam returned an unexpected response structure.",
                outcome="unexpected_structure",
            ) from exc

        logger.info(
            "steam_search_completed status=ok duration_ms=%s candidate_count=%s "
            "invalid_count=%s truncated_count=%s",
            duration_ms,
            len(result.candidates),
            result.invalid_count,
            result.truncated_count,
        )

        return result

    # ── Cooldown ───────────────────────────────────────────────────────

    def _enforce_cooldown(self, key: str) -> None:
        """Raise when the same term was queried within the cooldown.

        The key is the encoded term, so two spellings of the same term
        that encode identically share the cooldown.
        """

        if self._cooldown_seconds <= 0:
            return

        now = time.monotonic()
        with self._cooldown_lock:
            last = self._last_query_at.get(key)
            if last is not None and (now - last) < self._cooldown_seconds:
                remaining = self._cooldown_seconds - (now - last)
                logger.warning(
                    "steam_search_cooldown_rejected remaining_ms=%s",
                    int(remaining * 1000),
                )
                raise SteamSearchCooldownError(
                    "This Steam search was made too recently.",
                    outcome="cooldown",
                )
            # The timestamp is recorded before the request is made, so a
            # failed request still consumes the window.
            self._last_query_at[key] = now

    def reset_cooldown(self) -> None:
        """Forget every recorded query timestamp."""

        with self._cooldown_lock:
            self._last_query_at.clear()

    # ── HTTP ───────────────────────────────────────────────────────────

    def _request_json(self, url: str) -> Any:
        """Perform the request and return the decoded JSON body."""

        try:
            with self._client.stream("GET", url) as response:
                status_code = response.status_code

                if status_code == 429:
                    # No automatic retry: the caller decides when to try
                    # again.
                    logger.warning("steam_search_rate_limited")
                    raise SteamRetrievalError(
                        "Steam is rate-limiting requests.", outcome="rate_limited"
                    )

                if status_code >= 400:
                    logger.warning(
                        "steam_search_http_error status=%s", status_code
                    )
                    raise SteamRetrievalError(
                        "Steam returned an error.", outcome="http_error"
                    )

                content_type = response.headers.get("content-type", "")
                if not _is_json_content_type(content_type):
                    logger.warning(
                        "steam_search_unexpected_content_type content_type=%s",
                        content_type.split(";")[0].strip(),
                    )
                    raise SteamRetrievalError(
                        "Steam returned an unexpected content type.",
                        outcome="unexpected_content_type",
                    )

                body = self._read_bounded(response)
        except httpx.TimeoutException as exc:
            # The exception may embed the URL; it is not propagated.
            logger.warning("steam_search_timeout")
            raise SteamRetrievalError(
                "Steam did not respond in time.", outcome="timeout"
            ) from exc
        except httpx.HTTPError as exc:
            logger.warning(
                "steam_search_transport_error error_type=%s",
                type(exc).__name__,
            )
            raise SteamRetrievalError(
                "Steam could not be reached.", outcome="transport_error"
            ) from exc

        try:
            payload = json.loads(body)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("steam_search_invalid_json")
            raise SteamRetrievalError(
                "Steam returned an unreadable response.", outcome="invalid_json"
            ) from exc

        return payload

    def _read_bounded(self, response: httpx.Response) -> bytes:
        """Read the body while enforcing the size cap.

        The body is consumed chunk by chunk and the running total is
        checked on every chunk, so an oversized response is *aborted*
        rather than buffered. A declared ``Content-Length`` above the
        cap is rejected before the body is read at all.
        """

        content_length = response.headers.get("content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                declared = None
            if declared is not None and declared > self._max_response_bytes:
                logger.warning(
                    "steam_search_response_too_large declared_bytes=%s", declared
                )
                raise SteamRetrievalError(
                    "The Steam response exceeded the size limit.",
                    outcome="response_too_large",
                )

        chunks: List[bytes] = []
        total = 0
        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > self._max_response_bytes:
                logger.warning(
                    "steam_search_response_too_large actual_bytes=%s", total
                )
                raise SteamRetrievalError(
                    "The Steam response exceeded the size limit.",
                    outcome="response_too_large",
                )
            chunks.append(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        """Close the underlying HTTP client if this retriever owns it."""

        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "SteamSearchRetriever":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


# ── Helpers ───────────────────────────────────────────────────────────────


def _is_json_content_type(content_type: str) -> bool:
    """Return ``True`` when the content type is an accepted JSON type."""

    normalised = content_type.split(";")[0].strip().casefold()
    return normalised in ACCEPTED_CONTENT_TYPES


__all__ = [
    "ACCEPTED_CONTENT_TYPES",
    "DEFAULT_MAX_QUERY_LENGTH",
    "DEFAULT_MAX_RESPONSE_BYTES",
    "DEFAULT_QUERY_COOLDOWN_SECONDS",
    "MAX_STEAM_SEARCH_CANDIDATES",
    "SteamSearchRetriever",
    "build_storesearch_url",
    "validate_search_term",
]
