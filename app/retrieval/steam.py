"""The Steam ``appdetails`` retriever.

This module is the only place in the service that knows the Steam HTTP
API exists. Everything else depends on the small result type it
returns.

Guardrails
----------
**One fixed endpoint.** The URL is assembled from a module constant and
a validated integer App ID. There is no way for a caller to supply a
host, a path or a URL, so the service can never be turned into a
general-purpose fetcher.

**Bounded in both time and size.** Separate connect and read timeouts
keep a stalled connection from consuming the whole request budget, and
the body is read *while* counting bytes so an oversized response is
aborted instead of buffered.

**No automatic retry.** A rate-limited or failed request is reported,
never retried: a silent retry loop would multiply the load on an
already-struggling upstream and would hide the failure from the caller.

**Nothing sensitive is logged.** Log lines carry the App ID, the HTTP
status, a duration and a short outcome code. The response body is never
logged, and it is never propagated into an error message.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from app.errors import (
    SteamNotFoundError,
    SteamRetrievalError,
)
from app.retrieval.steam_mapper import (
    STEAM_APPDETAILS_ENDPOINT,
    STEAM_COUNTRY,
    STEAM_LOCALE,
    STEAM_LOCALE_ENGLISH,
    map_appdetails,
    titles_match,
)
from app.retrieval.steam_models import (
    IdentityCheck,
    SteamRetrievalResult,
)

logger = logging.getLogger(__name__)


#: The maximum App ID Steam assigns. Used as an upper bound so a
#: nonsensical identifier is rejected before a request is made.
MAX_STEAM_APP_ID = 2_147_483_647

#: The content types the retriever accepts. Steam answers
#: ``application/json``; anything else is rejected before parsing.
ACCEPTED_CONTENT_TYPES = ("application/json", "text/json")

#: The maximum number of bytes read from the Steam endpoint.
DEFAULT_MAX_RESPONSE_BYTES = 512 * 1024


def build_appdetails_url(app_id: int, *, locale: Optional[str] = None) -> str:
    """Return the request URL for one App ID.

    The URL is built from the fixed endpoint constant and the validated
    integer App ID. It is the *only* URL this module can produce.

    ``locale`` names the Steam locale. It defaults to
    :data:`STEAM_LOCALE` (the Vietnamese locale), so the default
    behaviour is unchanged. The translation path passes
    :data:`STEAM_LOCALE_ENGLISH`, because it needs the English store
    description to translate.
    """

    effective_locale = locale or STEAM_LOCALE
    return (
        f"{STEAM_APPDETAILS_ENDPOINT}"
        f"?appids={int(app_id)}"
        f"&l={effective_locale}"
        f"&cc={STEAM_COUNTRY}"
    )


def validate_app_id(app_id: Any) -> int:
    """Validate an App ID and return it as an ``int``.

    Raises :class:`ValueError` for anything that is not a positive
    integer within Steam's range. ``bool`` is rejected explicitly: it
    is an ``int`` subclass in Python but is never a valid App ID.
    """

    if isinstance(app_id, bool):
        raise ValueError("steam app id must be an integer")
    if isinstance(app_id, str):
        # A numeric string is accepted because JSON clients sometimes
        # serialise integers as strings, but nothing else is.
        text = app_id.strip()
        if not text.isdigit():
            raise ValueError("steam app id must be an integer")
        app_id = int(text)
    if not isinstance(app_id, int):
        raise ValueError("steam app id must be an integer")
    if app_id < 1 or app_id > MAX_STEAM_APP_ID:
        raise ValueError("steam app id is out of range")
    return app_id


class SteamAppIdRetriever:
    """Fetch and map one Steam app's public metadata.

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
        client: Optional[httpx.Client] = None,
        locale: str = STEAM_LOCALE,
    ) -> None:
        if connect_timeout_seconds <= 0:
            raise ValueError("connect_timeout_seconds must be positive")
        if read_timeout_seconds <= 0:
            raise ValueError("read_timeout_seconds must be positive")
        if max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")
        if not locale:
            raise ValueError("locale must not be empty")

        self._connect_timeout = float(connect_timeout_seconds)
        self._read_timeout = float(read_timeout_seconds)
        self._max_response_bytes = int(max_response_bytes)
        # The Steam locale every request is made with. It defaults to the
        # Vietnamese locale, so the default behaviour is unchanged; the
        # translation path constructs its own retriever with the English
        # locale.
        self._locale = locale

        # A reusable client keeps the connection warm. Tests inject
        # their own transport instead of touching the socket layer.
        self._client = client if client is not None else httpx.Client(
            timeout=httpx.Timeout(
                self._read_timeout, connect=self._connect_timeout
            ),
            follow_redirects=False,
        )
        self._owns_client = client is None

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def locale(self) -> str:
        """The Steam locale every request is made with."""

        return self._locale

    def fetch_payload(self, app_id: Any) -> Any:
        """Perform the request and return the raw decoded payload.

        This is the same guarded request :meth:`fetch` performs - the
        same URL, the same timeouts, the same size cap, the same
        content-type check, the same "no automatic retry" rule. It
        exists so a caller that needs a field the mapper does not
        model (the English description fields, for example) can read it
        **without** re-implementing the transport and without weakening
        any of the guards.
        """

        validated = validate_app_id(app_id)
        url = build_appdetails_url(validated, locale=self._locale)
        return self._request_json(url, validated)

    def fetch(
        self,
        app_id: Any,
        *,
        product_name: str,
        retrieved_at: str,
    ) -> SteamRetrievalResult:
        """Retrieve and map the app identified by ``app_id``.

        ``product_name`` is used **only** to compare the retrieved title
        with the admin's intended game. It is never used to look up an
        App ID.
        """

        validated = validate_app_id(app_id)
        url = build_appdetails_url(validated, locale=self._locale)

        started = time.monotonic()
        payload = self._request_json(url, validated)
        duration_ms = int((time.monotonic() - started) * 1000)

        facts, evidence, evidence_by_field = map_appdetails(
            validated, payload, retrieved_at=retrieved_at, locale=self._locale
        )

        matched, reason = titles_match(product_name, facts.name)
        identity = IdentityCheck(
            matched=matched,
            requested_name=product_name,
            retrieved_name=facts.name,
            reason=reason,
        )

        logger.info(
            "steam_retrieval_completed app_id=%s status=ok duration_ms=%s "
            "evidence_count=%s identity=%s",
            validated,
            duration_ms,
            len(evidence),
            reason,
        )

        return SteamRetrievalResult(
            app_id=validated,
            facts=facts,
            identity=identity,
            evidence=evidence,
            evidence_by_field=evidence_by_field,
        )

    # ── HTTP ───────────────────────────────────────────────────────────

    def _request_json(self, url: str, app_id: int) -> Any:
        """Perform the request and return the decoded JSON body."""

        try:
            with self._client.stream("GET", url) as response:
                status_code = response.status_code

                if status_code == 429:
                    # No automatic retry: the caller decides when to try
                    # again.
                    logger.warning("steam_retrieval_rate_limited app_id=%s", app_id)
                    raise SteamRetrievalError(
                        "Steam is rate-limiting requests.", outcome="rate_limited"
                    )

                if status_code >= 400:
                    logger.warning(
                        "steam_retrieval_http_error app_id=%s status=%s",
                        app_id,
                        status_code,
                    )
                    raise SteamRetrievalError(
                        "Steam returned an error.", outcome="http_error"
                    )

                content_type = response.headers.get("content-type", "")
                if not _is_json_content_type(content_type):
                    logger.warning(
                        "steam_retrieval_unexpected_content_type app_id=%s "
                        "content_type=%s",
                        app_id,
                        content_type.split(";")[0].strip(),
                    )
                    raise SteamRetrievalError(
                        "Steam returned an unexpected content type.",
                        outcome="unexpected_content_type",
                    )

                body = self._read_bounded(response, app_id)
        except httpx.TimeoutException as exc:
            # The exception may embed the URL; it is not propagated.
            logger.warning("steam_retrieval_timeout app_id=%s", app_id)
            raise SteamRetrievalError(
                "Steam did not respond in time.", outcome="timeout"
            ) from exc
        except httpx.HTTPError as exc:
            logger.warning(
                "steam_retrieval_transport_error app_id=%s error_type=%s",
                app_id,
                type(exc).__name__,
            )
            raise SteamRetrievalError(
                "Steam could not be reached.", outcome="transport_error"
            ) from exc

        try:
            payload = json.loads(body)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("steam_retrieval_invalid_json app_id=%s", app_id)
            raise SteamRetrievalError(
                "Steam returned an unreadable response.", outcome="invalid_json"
            ) from exc

        if not isinstance(payload, dict):
            logger.warning("steam_retrieval_unexpected_structure app_id=%s", app_id)
            raise SteamRetrievalError(
                "Steam returned an unexpected response structure.",
                outcome="unexpected_structure",
            )

        if not _payload_has_successful_entry(payload, app_id):
            logger.warning("steam_retrieval_not_found app_id=%s", app_id)
            raise SteamNotFoundError(
                "The Steam App ID does not identify a retrievable app.",
                outcome="not_found",
            )

        return payload

    def _read_bounded(self, response: httpx.Response, app_id: int) -> bytes:
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
                    "steam_retrieval_response_too_large app_id=%s declared_bytes=%s",
                    app_id,
                    declared,
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
                    "steam_retrieval_response_too_large app_id=%s actual_bytes=%s",
                    app_id,
                    total,
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

    def __enter__(self) -> "SteamAppIdRetriever":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


# ── Helpers ───────────────────────────────────────────────────────────────


def _is_json_content_type(content_type: str) -> bool:
    """Return ``True`` when the content type is an accepted JSON type."""

    normalised = content_type.split(";")[0].strip().casefold()
    return normalised in ACCEPTED_CONTENT_TYPES


def _payload_has_successful_entry(payload: Dict[str, Any], app_id: int) -> bool:
    """Return ``True`` when the payload holds usable ``data`` for the App ID."""

    entry = payload.get(str(app_id))
    if not isinstance(entry, dict):
        # Tolerate a differently-keyed payload only when it is
        # unambiguous: exactly one entry with a ``data`` object.
        candidates = [
            value
            for value in payload.values()
            if isinstance(value, dict) and isinstance(value.get("data"), dict)
        ]
        if len(candidates) != 1:
            return False
        entry = candidates[0]

    if entry.get("success") is not True:
        return False
    return isinstance(entry.get("data"), dict)


__all__ = [
    "ACCEPTED_CONTENT_TYPES",
    "DEFAULT_MAX_RESPONSE_BYTES",
    "MAX_STEAM_APP_ID",
    "SteamAppIdRetriever",
    "build_appdetails_url",
    "validate_app_id",
]
