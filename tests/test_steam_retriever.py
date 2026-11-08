"""Tests for the Steam retriever.

Every test runs fully offline: the transport is mocked with
``httpx.MockTransport``, so no request ever leaves the process. The
tests pin the guardrails the architecture requires:

* exactly one allow-listed endpoint is ever contacted,
* strict timeouts and a bounded body are enforced,
* the content type is checked before the body is parsed,
* every failure mode produces a safe domain error,
* nothing is retried automatically,
* no raw response body reaches an error message or a log line.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import httpx
import pytest

from app.errors import SteamNotFoundError, SteamRetrievalError
from app.retrieval.steam import (
    MAX_STEAM_APP_ID,
    SteamAppIdRetriever,
    build_appdetails_url,
    validate_app_id,
)


FIXTURES = Path(__file__).parent / "fixtures" / "steam"


def _load(name: str) -> Dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _retriever(
    handler: Any,
    *,
    max_response_bytes: int = 512 * 1024,
) -> SteamAppIdRetriever:
    """Build a retriever whose transport is fully mocked."""

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    return SteamAppIdRetriever(
        connect_timeout_seconds=1.0,
        read_timeout_seconds=1.0,
        max_response_bytes=max_response_bytes,
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


# ── URL construction ──────────────────────────────────────────────────────


class TestUrlConstruction:
    """The endpoint is fixed; only the App ID varies."""

    def test_url_uses_the_allowlisted_endpoint(self) -> None:
        url = build_appdetails_url(1245620)
        assert url.startswith("https://store.steampowered.com/api/appdetails")

    def test_url_carries_the_app_id_and_fixed_parameters(self) -> None:
        url = build_appdetails_url(1245620)
        assert "appids=1245620" in url
        assert "l=vietnamese" in url
        assert "cc=vn" in url

    def test_url_does_not_accept_a_host(self) -> None:
        # The builder takes an integer, so a host can never be injected.
        with pytest.raises((ValueError, TypeError)):
            build_appdetails_url("https://evil.example.com")  # type: ignore[arg-type]

    def test_request_targets_the_allowlisted_host(self) -> None:
        seen: Dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            return httpx.Response(
                200,
                content=json.dumps(_load("appdetails_success_full.json")).encode(),
                headers={"content-type": "application/json"},
            )

        retriever = _retriever(handler)
        retriever.fetch(
            1245620, product_name="ELDEN RING", retrieved_at="2026-01-01T00:00:00+00:00"
        )
        assert seen["url"].startswith(
            "https://store.steampowered.com/api/appdetails?appids=1245620"
        )


# ── App ID validation ─────────────────────────────────────────────────────


class TestAppIdValidation:
    """Only a positive, in-range integer is a usable App ID."""

    @pytest.mark.parametrize("value", [1, 1245620, MAX_STEAM_APP_ID])
    def test_valid_ids_are_accepted(self, value: int) -> None:
        assert validate_app_id(value) == value

    @pytest.mark.parametrize(
        "value", [0, -1, MAX_STEAM_APP_ID + 1, 1.5, "abc", "", None, True, False]
    )
    def test_invalid_ids_are_rejected(self, value: Any) -> None:
        with pytest.raises(ValueError):
            validate_app_id(value)

    def test_numeric_string_is_accepted(self) -> None:
        assert validate_app_id(" 1245620 ") == 1245620


# ── Successful retrieval ──────────────────────────────────────────────────


class TestSuccessfulRetrieval:
    """A well-formed response is parsed into facts and evidence."""

    def test_full_payload_is_parsed(self) -> None:
        payload = _load("appdetails_success_full.json")
        retriever = _retriever(_json_handler(payload))
        result = retriever.fetch(
            1245620, product_name="ELDEN RING", retrieved_at="2026-01-01T00:00:00+00:00"
        )
        assert result.facts.name == "ELDEN RING"
        assert result.facts.release_date == "2022-02-25"
        assert result.facts.platforms == ["Windows"]
        assert result.facts.minimum.os == "Windows 10 64-bit"
        assert result.facts.minimum.memory == "12 GB RAM"
        assert result.facts.recommended.processor == "Intel Core i7-6700K"

    def test_service_owned_evidence_is_created(self) -> None:
        payload = _load("appdetails_success_full.json")
        retriever = _retriever(_json_handler(payload))
        result = retriever.fetch(
            1245620, product_name="ELDEN RING", retrieved_at="2026-01-01T00:00:00+00:00"
        )
        assert result.evidence
        for item in result.evidence:
            assert item.source_type == "steam"
            assert item.evidence_id.startswith("steam:1245620:")
            assert item.retrieved_at == "2026-01-01T00:00:00+00:00"
            assert item.source_ref is not None
            assert "appids=1245620" in item.source_ref

    def test_evidence_ids_are_unique(self) -> None:
        payload = _load("appdetails_success_full.json")
        retriever = _retriever(_json_handler(payload))
        result = retriever.fetch(
            1245620, product_name="ELDEN RING", retrieved_at="2026-01-01T00:00:00+00:00"
        )
        ids = [item.evidence_id for item in result.evidence]
        assert len(ids) == len(set(ids))

    def test_evidence_is_mapped_to_field_paths(self) -> None:
        payload = _load("appdetails_success_full.json")
        retriever = _retriever(_json_handler(payload))
        result = retriever.fetch(
            1245620, product_name="ELDEN RING", retrieved_at="2026-01-01T00:00:00+00:00"
        )
        assert result.evidence_id_for("release_date") is not None
        assert result.evidence_id_for("platforms") is not None
        assert result.evidence_id_for("system_requirements.minimum.os") is not None
        assert result.evidence_id_for("system_requirements.recommended.os") is not None

    def test_minimal_payload_parses(self) -> None:
        payload = _load("appdetails_success_minimal.json")
        retriever = _retriever(_json_handler(payload))
        result = retriever.fetch(
            1245620,
            product_name="Aurora's Echo",
            retrieved_at="2026-01-01T00:00:00+00:00",
        )
        assert result.facts.name == "Aurora's Echo"
        # An unreleased game has no parsable date: the field stays
        # unknown rather than becoming a wrong value.
        assert result.facts.release_date is None
        assert result.facts.platforms == ["Windows"]
        assert result.facts.minimum.os is None


# ── Failure modes ─────────────────────────────────────────────────────────


class TestFailureModes:
    """Every failure is a safe domain error, never a raw exception."""

    def test_success_false_raises_not_found(self) -> None:
        payload = _load("appdetails_success_false.json")
        retriever = _retriever(_json_handler(payload))
        with pytest.raises(SteamNotFoundError):
            retriever.fetch(
                1245620,
                product_name="ELDEN RING",
                retrieved_at="2026-01-01T00:00:00+00:00",
            )

    def test_missing_data_key_raises_not_found(self) -> None:
        retriever = _retriever(_json_handler({"1245620": {"success": True}}))
        with pytest.raises(SteamNotFoundError):
            retriever.fetch(
                1245620,
                product_name="ELDEN RING",
                retrieved_at="2026-01-01T00:00:00+00:00",
            )

    def test_invalid_json_raises_retrieval_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=b"{not json",
                headers={"content-type": "application/json"},
            )

        retriever = _retriever(handler)
        with pytest.raises(SteamRetrievalError):
            retriever.fetch(
                1245620,
                product_name="ELDEN RING",
                retrieved_at="2026-01-01T00:00:00+00:00",
            )

    def test_unexpected_structure_raises_retrieval_error(self) -> None:
        retriever = _retriever(_json_handler([1, 2, 3]))
        with pytest.raises(SteamRetrievalError):
            retriever.fetch(
                1245620,
                product_name="ELDEN RING",
                retrieved_at="2026-01-01T00:00:00+00:00",
            )

    def test_non_json_content_type_is_rejected(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=b"<html>not json</html>",
                headers={"content-type": "text/html; charset=utf-8"},
            )

        retriever = _retriever(handler)
        with pytest.raises(SteamRetrievalError) as excinfo:
            retriever.fetch(
                1245620,
                product_name="ELDEN RING",
                retrieved_at="2026-01-01T00:00:00+00:00",
            )
        assert excinfo.value.outcome == "unexpected_content_type"

    def test_rate_limit_raises_retrieval_error(self) -> None:
        retriever = _retriever(_json_handler({}, status_code=429))
        with pytest.raises(SteamRetrievalError) as excinfo:
            retriever.fetch(
                1245620,
                product_name="ELDEN RING",
                retrieved_at="2026-01-01T00:00:00+00:00",
            )
        assert excinfo.value.outcome == "rate_limited"

    def test_server_error_raises_retrieval_error(self) -> None:
        retriever = _retriever(_json_handler({}, status_code=503))
        with pytest.raises(SteamRetrievalError) as excinfo:
            retriever.fetch(
                1245620,
                product_name="ELDEN RING",
                retrieved_at="2026-01-01T00:00:00+00:00",
            )
        assert excinfo.value.outcome == "http_error"

    def test_connect_error_raises_retrieval_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        retriever = _retriever(handler)
        with pytest.raises(SteamRetrievalError) as excinfo:
            retriever.fetch(
                1245620,
                product_name="ELDEN RING",
                retrieved_at="2026-01-01T00:00:00+00:00",
            )
        assert excinfo.value.outcome == "transport_error"

    def test_read_timeout_raises_retrieval_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        retriever = _retriever(handler)
        with pytest.raises(SteamRetrievalError) as excinfo:
            retriever.fetch(
                1245620,
                product_name="ELDEN RING",
                retrieved_at="2026-01-01T00:00:00+00:00",
            )
        assert excinfo.value.outcome == "timeout"

    def test_oversized_response_is_rejected(self) -> None:
        payload = _load("appdetails_success_full.json")
        # The cap is set below the fixture's size, so the body is
        # aborted while it is being read.
        retriever = _retriever(_json_handler(payload), max_response_bytes=512)
        with pytest.raises(SteamRetrievalError) as excinfo:
            retriever.fetch(
                1245620,
                product_name="ELDEN RING",
                retrieved_at="2026-01-01T00:00:00+00:00",
            )
        assert excinfo.value.outcome == "response_too_large"

    def test_oversized_declared_length_is_rejected_before_reading(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=b"{}",
                headers={
                    "content-type": "application/json",
                    "content-length": "99999999",
                },
            )

        retriever = _retriever(handler, max_response_bytes=1024)
        with pytest.raises(SteamRetrievalError) as excinfo:
            retriever.fetch(
                1245620,
                product_name="ELDEN RING",
                retrieved_at="2026-01-01T00:00:00+00:00",
            )
        assert excinfo.value.outcome == "response_too_large"


# ── No retry, no leakage ──────────────────────────────────────────────────


class TestNoRetryAndNoLeakage:
    """The retriever neither retries nor leaks the response body."""

    def test_rate_limited_request_is_not_retried(self) -> None:
        calls = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            return httpx.Response(
                429, content=b"{}", headers={"content-type": "application/json"}
            )

        retriever = _retriever(handler)
        with pytest.raises(SteamRetrievalError):
            retriever.fetch(
                1245620,
                product_name="ELDEN RING",
                retrieved_at="2026-01-01T00:00:00+00:00",
            )
        assert calls["count"] == 1

    def test_failed_request_is_not_retried(self) -> None:
        calls = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            raise httpx.ConnectError("boom", request=request)

        retriever = _retriever(handler)
        with pytest.raises(SteamRetrievalError):
            retriever.fetch(
                1245620,
                product_name="ELDEN RING",
                retrieved_at="2026-01-01T00:00:00+00:00",
            )
        assert calls["count"] == 1

    def test_error_message_contains_no_response_body(self) -> None:
        secret = "SUPER-SECRET-BODY-CONTENT"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=secret.encode("utf-8"),
                headers={"content-type": "application/json"},
            )

        retriever = _retriever(handler)
        with pytest.raises(SteamRetrievalError) as excinfo:
            retriever.fetch(
                1245620,
                product_name="ELDEN RING",
                retrieved_at="2026-01-01T00:00:00+00:00",
            )
        assert secret not in str(excinfo.value)
        assert secret not in str(excinfo.value.message)

    def test_log_output_contains_no_response_body(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        secret = "SUPER-SECRET-BODY-CONTENT"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=secret.encode("utf-8"),
                headers={"content-type": "application/json"},
            )

        retriever = _retriever(handler)
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(SteamRetrievalError):
                retriever.fetch(
                    1245620,
                    product_name="ELDEN RING",
                    retrieved_at="2026-01-01T00:00:00+00:00",
                )
        assert secret not in caplog.text

    def test_log_output_contains_the_app_id(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        retriever = _retriever(_json_handler(_load("appdetails_success_full.json")))
        with caplog.at_level(logging.INFO):
            retriever.fetch(
                1245620,
                product_name="ELDEN RING",
                retrieved_at="2026-01-01T00:00:00+00:00",
            )
        assert "1245620" in caplog.text
