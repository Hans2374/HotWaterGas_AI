"""Tests for the Steam Store Search core.

Every test runs fully offline: the transport is mocked with
``httpx.MockTransport``, so no request ever leaves the process. The
captured fixture
(``tests/fixtures/steam/storesearch_portal_2.json``) pins the real
response shape; the synthetic payloads cover the failure modes.

The tests pin the guardrails the design requires:

* exactly one allow-listed endpoint is ever contacted,
* the term is validated and URL-encoded, never used to build a host,
* strict timeouts and a bounded body are enforced,
* the content type is checked before the body is parsed,
* every failure mode produces a safe domain error,
* nothing is retried automatically,
* the repeated-query cooldown is enforced,
* no search term and no raw response body reaches a log line.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pytest

from app.errors import (
    SteamRetrievalError,
    SteamSearchCooldownError,
    SteamSearchQueryInvalidError,
)
from app.retrieval.steam_search import (
    DEFAULT_MAX_QUERY_LENGTH,
    SteamSearchRetriever,
    build_storesearch_url,
    validate_search_term,
)
from app.retrieval.steam_search_mapper import (
    MAX_STEAM_SEARCH_CANDIDATES,
    STEAM_STORESEARCH_ENDPOINT,
    SteamSearchPayloadError,
    map_storesearch,
)


FIXTURES = Path(__file__).parent / "fixtures" / "steam"


def _load(name: str) -> Dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _retriever(
    handler: Any,
    *,
    max_response_bytes: int = 512 * 1024,
    max_candidates: int = MAX_STEAM_SEARCH_CANDIDATES,
    cooldown_seconds: float = 0.0,
) -> SteamSearchRetriever:
    """Build a retriever whose transport is fully mocked."""

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    return SteamSearchRetriever(
        connect_timeout_seconds=1.0,
        read_timeout_seconds=1.0,
        max_response_bytes=max_response_bytes,
        max_candidates=max_candidates,
        cooldown_seconds=cooldown_seconds,
        client=client,
    )


def _json_handler(
    payload: Any,
    *,
    status_code: int = 200,
    content_type: str = "application/json",
) -> Any:
    """Return a transport handler that answers with a JSON body."""

    body = json.dumps(payload).encode("utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            content=body,
            headers={"content-type": content_type},
        )

    return handler


def _fixture_handler() -> Any:
    """Return a handler that answers with the captured fixture."""

    payload = _load("storesearch_portal_2.json")
    return _json_handler(payload)


# ── URL construction ──────────────────────────────────────────────────────


class TestUrlConstruction:
    """The endpoint is fixed; only the term varies."""

    def test_url_uses_the_allowlisted_endpoint(self) -> None:
        url = build_storesearch_url("Portal%202")
        assert url.startswith(STEAM_STORESEARCH_ENDPOINT)
        assert STEAM_STORESEARCH_ENDPOINT == (
            "https://store.steampowered.com/api/storesearch/"
        )

    def test_url_carries_the_term_and_fixed_parameters(self) -> None:
        url = build_storesearch_url("Portal%202")
        assert "term=Portal%202" in url
        assert "l=vietnamese" in url
        assert "cc=vn" in url

    def test_request_targets_the_allowlisted_host(self) -> None:
        seen: Dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            return httpx.Response(
                200,
                content=json.dumps(_load("storesearch_portal_2.json")).encode(),
                headers={"content-type": "application/json"},
            )

        retriever = _retriever(handler)
        retriever.search("Portal 2")
        assert seen["url"].startswith(
            "https://store.steampowered.com/api/storesearch/?term=Portal%202"
        )
        assert "l=vietnamese" in seen["url"]
        assert "cc=vn" in seen["url"]

    def test_term_cannot_inject_a_host(self) -> None:
        # The term is percent-encoded, so it can never become a host or
        # a second query parameter.
        encoded = validate_search_term("evil.example.com/?x=1")
        assert "/" not in encoded
        assert "?" not in encoded
        assert encoded.startswith("evil.example.com%2F%3Fx%3D1")

    def test_a_url_shaped_term_still_targets_the_allowlisted_host(self) -> None:
        seen: Dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            return httpx.Response(
                200,
                content=json.dumps({"total": 0, "items": []}).encode(),
                headers={"content-type": "application/json"},
            )

        retriever = _retriever(handler)
        retriever.search("https://evil.example.com/api/storesearch/")
        assert seen["url"].startswith(
            "https://store.steampowered.com/api/storesearch/"
        )
        assert "evil.example.com" not in seen["url"].split("?")[0]


# ── Term validation ───────────────────────────────────────────────────────


class TestTermValidation:
    """The term is trimmed, bounded and non-empty."""

    def test_term_is_trimmed(self) -> None:
        assert validate_search_term("  Portal 2  ") == "Portal%202"

    def test_empty_term_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            validate_search_term("   ")

    def test_non_string_term_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            validate_search_term(None)  # type: ignore[arg-type]

    def test_excessively_long_term_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            validate_search_term("x" * (DEFAULT_MAX_QUERY_LENGTH + 1))

    def test_maximum_length_term_is_accepted(self) -> None:
        term = "x" * DEFAULT_MAX_QUERY_LENGTH
        assert validate_search_term(term) == "x" * DEFAULT_MAX_QUERY_LENGTH

    def test_invalid_term_raises_the_domain_error(self) -> None:
        retriever = _retriever(_fixture_handler())
        with pytest.raises(SteamSearchQueryInvalidError) as excinfo:
            retriever.search("   ")
        assert excinfo.value.outcome == "invalid_query"
        assert excinfo.value.http_status == 422


# ── Mapping the captured fixture ──────────────────────────────────────────


class TestFixtureMapping:
    """The captured fixture is the reference response shape."""

    def test_fixture_maps_to_ten_candidates(self) -> None:
        result = map_storesearch(_load("storesearch_portal_2.json"))
        assert result.total == 10
        assert len(result.candidates) == 10
        assert result.invalid_count == 0
        assert result.truncated_count == 0

    def test_first_candidate_matches_the_fixture(self) -> None:
        result = map_storesearch(_load("storesearch_portal_2.json"))
        first = result.candidates[0]
        assert first.app_id == 620
        assert first.name == "Portal 2"
        assert first.app_type == "app"
        assert first.tiny_image.startswith("https://")
        assert first.metascore == 95
        assert first.platforms == ["Windows", "Linux"]
        assert first.streamingvideo is False
        assert first.controller_support == "full"
        assert first.price is not None
        assert first.price.currency == "VND"
        assert first.price.final == 14000000.0

    def test_candidate_without_price_has_none(self) -> None:
        result = map_storesearch(_load("storesearch_portal_2.json"))
        soundtrack = next(
            c for c in result.candidates if c.app_id == 323180
        )
        assert soundtrack.price is None
        assert soundtrack.metascore is None
        assert soundtrack.controller_support is None

    def test_empty_metascore_string_maps_to_none(self) -> None:
        result = map_storesearch(_load("storesearch_portal_2.json"))
        demo = next(c for c in result.candidates if c.app_id == 3516070)
        assert demo.metascore is None

    def test_search_results_are_not_type_filtered(self) -> None:
        # The fixture contains a soundtrack and a demo, both reported as
        # type "app". The search layer must not drop them: product-type
        # validation belongs to the later appdetails stage.
        result = map_storesearch(_load("storesearch_portal_2.json"))
        names = [c.name for c in result.candidates]
        assert "Portal 2 Soundtrack" in names
        assert "Grand Emprise 2: Portals Apart Demo" in names

    def test_retriever_returns_the_fixture(self) -> None:
        retriever = _retriever(_fixture_handler())
        result = retriever.search("Portal 2")
        assert result.total == 10
        assert result.candidates[0].app_id == 620


# ── Optional fields and edge payloads ─────────────────────────────────────


class TestOptionalFields:
    """Optional fields are optional; required ones are enforced."""

    def test_item_with_only_id_and_name_is_valid(self) -> None:
        result = map_storesearch(
            {"total": 1, "items": [{"id": 10, "name": "Solo Game"}]}
        )
        assert len(result.candidates) == 1
        candidate = result.candidates[0]
        assert candidate.app_id == 10
        assert candidate.name == "Solo Game"
        assert candidate.price is None
        assert candidate.metascore is None
        assert candidate.platforms == []
        assert candidate.tiny_image is None
        assert candidate.app_type is None

    def test_empty_items_list_maps_to_no_candidates(self) -> None:
        result = map_storesearch({"total": 0, "items": []})
        assert result.total == 0
        assert result.candidates == []
        assert result.invalid_count == 0

    def test_missing_total_falls_back_to_the_item_count(self) -> None:
        result = map_storesearch({"items": [{"id": 1, "name": "A"}]})
        assert result.total == 1

    def test_non_http_image_url_is_dropped(self) -> None:
        result = map_storesearch(
            {
                "total": 1,
                "items": [
                    {"id": 1, "name": "A", "tiny_image": "javascript:alert(1)"}
                ],
            }
        )
        assert result.candidates[0].tiny_image is None


# ── Invalid payloads ──────────────────────────────────────────────────────


class TestInvalidPayloads:
    """A malformed payload becomes a domain error, never a crash."""

    @pytest.mark.parametrize(
        "payload",
        [
            [],
            "text",
            42,
            None,
            {},
            {"items": {}},
            {"items": "nope"},
            {"total": 3},
        ],
    )
    def test_invalid_top_level_payload_raises(self, payload: Any) -> None:
        with pytest.raises(SteamSearchPayloadError):
            map_storesearch(payload)

    def test_invalid_top_level_payload_becomes_a_domain_error(self) -> None:
        retriever = _retriever(_json_handler({"nope": True}))
        with pytest.raises(SteamRetrievalError) as excinfo:
            retriever.search("Portal 2")
        assert excinfo.value.outcome == "unexpected_structure"

    def test_non_object_item_is_skipped(self) -> None:
        result = map_storesearch(
            {"total": 2, "items": ["nope", {"id": 5, "name": "Real"}]}
        )
        assert [c.app_id for c in result.candidates] == [5]
        assert result.invalid_count == 1

    def test_item_without_id_is_skipped(self) -> None:
        result = map_storesearch({"total": 1, "items": [{"name": "No Id"}]})
        assert result.candidates == []
        assert result.invalid_count == 1

    def test_item_with_zero_id_is_skipped(self) -> None:
        result = map_storesearch({"total": 1, "items": [{"id": 0, "name": "X"}]})
        assert result.candidates == []
        assert result.invalid_count == 1

    def test_item_with_negative_id_is_skipped(self) -> None:
        result = map_storesearch({"total": 1, "items": [{"id": -3, "name": "X"}]})
        assert result.candidates == []
        assert result.invalid_count == 1

    def test_item_with_boolean_id_is_skipped(self) -> None:
        result = map_storesearch({"total": 1, "items": [{"id": True, "name": "X"}]})
        assert result.candidates == []
        assert result.invalid_count == 1

    def test_item_with_empty_name_is_skipped(self) -> None:
        result = map_storesearch({"total": 1, "items": [{"id": 7, "name": "  "}]})
        assert result.candidates == []
        assert result.invalid_count == 1

    def test_item_with_non_string_name_is_skipped(self) -> None:
        result = map_storesearch({"total": 1, "items": [{"id": 7, "name": 42}]})
        assert result.candidates == []
        assert result.invalid_count == 1

    def test_one_invalid_item_does_not_lose_the_valid_ones(self) -> None:
        result = map_storesearch(
            {
                "total": 3,
                "items": [
                    {"id": 1, "name": "Good"},
                    {"id": "abc", "name": "Bad"},
                    {"id": 3, "name": "Also Good"},
                ],
            }
        )
        assert [c.app_id for c in result.candidates] == [1, 3]
        assert result.invalid_count == 1


# ── Duplicates and the candidate limit ────────────────────────────────────


class TestDuplicatesAndLimit:
    """Duplicates collapse; the candidate cap is hard."""

    def test_duplicate_ids_are_collapsed(self) -> None:
        result = map_storesearch(
            {
                "total": 2,
                "items": [
                    {"id": 620, "name": "Portal 2"},
                    {"id": 620, "name": "Portal 2"},
                ],
            }
        )
        assert len(result.candidates) == 1
        assert result.candidates[0].name == "Portal 2"

    def test_candidate_limit_is_fifteen(self) -> None:
        items = [{"id": i, "name": f"Game {i}"} for i in range(1, 26)]
        result = map_storesearch({"total": len(items), "items": items})
        assert len(result.candidates) == MAX_STEAM_SEARCH_CANDIDATES
        assert MAX_STEAM_SEARCH_CANDIDATES == 15
        assert result.truncated_count == 10

    def test_candidate_limit_is_applied_by_the_retriever(self) -> None:
        items = [{"id": i, "name": f"Game {i}"} for i in range(1, 26)]
        retriever = _retriever(_json_handler({"total": len(items), "items": items}))
        result = retriever.search("Game")
        assert len(result.candidates) == 15

    def test_limit_keeps_the_first_candidates(self) -> None:
        items = [{"id": i, "name": f"Game {i}"} for i in range(1, 21)]
        result = map_storesearch({"total": len(items), "items": items})
        assert result.candidates[0].app_id == 1
        assert result.candidates[-1].app_id == 15


# ── Upstream failures ─────────────────────────────────────────────────────


class TestUpstreamFailures:
    """Every upstream failure becomes a safe domain error."""

    def test_http_error_status(self) -> None:
        retriever = _retriever(
            _json_handler({}, status_code=500, content_type="application/json")
        )
        with pytest.raises(SteamRetrievalError) as excinfo:
            retriever.search("Portal 2")
        assert excinfo.value.outcome == "http_error"

    def test_rate_limit_status(self) -> None:
        retriever = _retriever(
            _json_handler({}, status_code=429, content_type="application/json")
        )
        with pytest.raises(SteamRetrievalError) as excinfo:
            retriever.search("Portal 2")
        assert excinfo.value.outcome == "rate_limited"

    def test_timeout(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        retriever = _retriever(handler)
        with pytest.raises(SteamRetrievalError) as excinfo:
            retriever.search("Portal 2")
        assert excinfo.value.outcome == "timeout"

    def test_transport_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        retriever = _retriever(handler)
        with pytest.raises(SteamRetrievalError) as excinfo:
            retriever.search("Portal 2")
        assert excinfo.value.outcome == "transport_error"

    def test_wrong_content_type(self) -> None:
        retriever = _retriever(
            _json_handler({"total": 0, "items": []}, content_type="text/html")
        )
        with pytest.raises(SteamRetrievalError) as excinfo:
            retriever.search("Portal 2")
        assert excinfo.value.outcome == "unexpected_content_type"

    def test_json_content_type_with_charset_is_accepted(self) -> None:
        retriever = _retriever(
            _json_handler(
                {"total": 0, "items": []},
                content_type="application/json; charset=utf-8",
            )
        )
        result = retriever.search("Portal 2")
        assert result.candidates == []

    def test_invalid_json_body(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=b"{not json",
                headers={"content-type": "application/json"},
            )

        retriever = _retriever(handler)
        with pytest.raises(SteamRetrievalError) as excinfo:
            retriever.search("Portal 2")
        assert excinfo.value.outcome == "invalid_json"

    def test_oversized_declared_body_is_rejected(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=b"{}",
                headers={"content-type": "application/json", "content-length": "999999"},
            )

        retriever = _retriever(handler, max_response_bytes=1024)
        with pytest.raises(SteamRetrievalError) as excinfo:
            retriever.search("Portal 2")
        assert excinfo.value.outcome == "response_too_large"

    def test_oversized_streamed_body_is_rejected(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=b"x" * 5000,
                headers={"content-type": "application/json"},
            )

        retriever = _retriever(handler, max_response_bytes=1024)
        with pytest.raises(SteamRetrievalError) as excinfo:
            retriever.search("Portal 2")
        assert excinfo.value.outcome == "response_too_large"

    def test_nothing_is_retried_automatically(self) -> None:
        calls: List[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(
                503,
                content=b"{}",
                headers={"content-type": "application/json"},
            )

        retriever = _retriever(handler)
        with pytest.raises(SteamRetrievalError):
            retriever.search("Portal 2")
        assert len(calls) == 1


# ── Cooldown ──────────────────────────────────────────────────────────────


class TestCooldown:
    """A repeated term is refused inside the cooldown window."""

    def test_repeated_query_within_the_window_is_refused(self) -> None:
        calls: List[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(
                200,
                content=json.dumps({"total": 0, "items": []}).encode(),
                headers={"content-type": "application/json"},
            )

        retriever = _retriever(handler, cooldown_seconds=2.0)
        retriever.search("Portal 2")
        with pytest.raises(SteamSearchCooldownError) as excinfo:
            retriever.search("Portal 2")
        assert excinfo.value.outcome == "cooldown"
        assert excinfo.value.http_status == 429
        assert len(calls) == 1

    def test_a_different_term_is_not_blocked(self) -> None:
        retriever = _retriever(_fixture_handler(), cooldown_seconds=2.0)
        retriever.search("Portal 2")
        result = retriever.search("Half-Life")
        assert result.total == 10

    def test_the_cooldown_expires(self) -> None:
        # The sleep is deliberately generous: the clock used by the
        # cooldown is a monotonic one whose resolution is coarse on
        # Windows, so a sleep that only just clears the window would
        # make this test fail intermittently.
        retriever = _retriever(_fixture_handler(), cooldown_seconds=0.05)
        retriever.search("Portal 2")
        import time as _time

        _time.sleep(0.25)
        result = retriever.search("Portal 2")
        assert result.total == 10

    def test_reset_clears_the_cooldown(self) -> None:
        retriever = _retriever(_fixture_handler(), cooldown_seconds=2.0)
        retriever.search("Portal 2")
        retriever.reset_cooldown()
        result = retriever.search("Portal 2")
        assert result.total == 10

    def test_the_cooldown_is_scoped_to_the_retriever_instance(self) -> None:
        first = _retriever(_fixture_handler(), cooldown_seconds=2.0)
        second = _retriever(_fixture_handler(), cooldown_seconds=2.0)
        first.search("Portal 2")
        # A second instance has its own registry: the cooldown is
        # process-local per retriever, not global.
        result = second.search("Portal 2")
        assert result.total == 10


# ── Logging hygiene ───────────────────────────────────────────────────────


class TestLoggingHygiene:
    """Neither the term nor the body reaches the log."""

    def test_the_term_is_not_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        retriever = _retriever(_fixture_handler())
        with caplog.at_level(logging.DEBUG):
            retriever.search("Secret Portal Title")
        assert "Secret Portal Title" not in caplog.text

    def test_the_response_body_is_not_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        retriever = _retriever(_fixture_handler())
        with caplog.at_level(logging.DEBUG):
            retriever.search("Portal 2")
        assert "tiny_image" not in caplog.text
        assert "akamai" not in caplog.text

    def test_an_upstream_error_message_does_not_leak_the_body(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                500,
                content=b'{"secret":"leak"}',
                headers={"content-type": "application/json"},
            )

        retriever = _retriever(handler)
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(SteamRetrievalError) as excinfo:
                retriever.search("Portal 2")
        assert "leak" not in caplog.text
        assert "leak" not in str(excinfo.value)


# ── Configuration guards ──────────────────────────────────────────────────


class TestConstructorGuards:
    """The constructor refuses nonsensical configuration."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"connect_timeout_seconds": 0},
            {"read_timeout_seconds": 0},
            {"max_response_bytes": 0},
            {"max_candidates": -1},
            {"cooldown_seconds": -1},
        ],
    )
    def test_invalid_configuration_is_rejected(self, kwargs: Dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            SteamSearchRetriever(**kwargs)  # type: ignore[arg-type]
