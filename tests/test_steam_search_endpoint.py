"""Tests for the internal Steam Store Search endpoint.

The tests build the **real** FastAPI app (so the OpenAPI contract is
verified) but replace the lifespan-built search retriever with a fake,
so no Steam request is ever made. They cover:

* Authentication: missing token, wrong token, correct token.
* The feature switch: disabled -> 503, and no upstream call.
* A successful search: the documented response shape.
* An empty result set: 200 with an empty candidate list.
* An invalid query: 422.
* The repeated-query cooldown: 429.
* An upstream rate limit: 429.
* An upstream failure: 502.
* The response excludes the raw Steam fields that are not needed to
  pick a candidate.
* The endpoint does not touch the evidence pipeline.
* The generation endpoints keep their contract.

Everything is offline. The session-wide guards in ``conftest.py``
block dotenv loading and any real network egress, so a forgotten mock
fails the test instead of reaching a live service.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pytest
from fastapi.testclient import TestClient

from app.dependencies.auth import HEADER_NAME
from app.errors import (
    SteamRetrievalError,
    SteamSearchCooldownError,
    SteamSearchQueryInvalidError,
)
from app.retrieval.steam_search_models import (
    SteamSearchCandidate,
    SteamSearchPrice,
    SteamSearchResult,
)


FIXTURES = Path(__file__).parent / "fixtures" / "steam"

#: A local, non-production token. It exists only inside the test
#: process and is never a real credential.
TEST_TOKEN = "phase-2-local-test-token"

ENDPOINT = "/internal/steam/search"


# ── Fakes ─────────────────────────────────────────────────────────────────


class FakeSearchRetriever:
    """A stand-in for ``SteamSearchRetriever``.

    It records every call so a test can prove that the endpoint made
    exactly one upstream call and never called ``appdetails``.
    """

    def __init__(self) -> None:
        self.calls: List[str] = []
        self.next_result: Optional[SteamSearchResult] = None
        self.next_error: Optional[Exception] = None

    def search(self, term: Any) -> SteamSearchResult:
        self.calls.append(term)
        if self.next_error is not None:
            raise self.next_error
        assert self.next_result is not None
        return self.next_result


def _candidate(
    app_id: int = 620,
    name: str = "Portal 2",
    *,
    app_type: Optional[str] = "app",
    tiny_image: Optional[str] = (
        "https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/620/"
        "capsule_231x87.jpg"
    ),
    price: Optional[SteamSearchPrice] = None,
    metascore: Optional[int] = 95,
) -> SteamSearchCandidate:
    return SteamSearchCandidate(
        app_id=app_id,
        name=name,
        app_type=app_type,
        tiny_image=tiny_image,
        price=price,
        metascore=metascore,
        platforms=["Windows", "Linux"],
        streamingvideo=False,
        controller_support="full",
    )


def _result(
    candidates: Optional[List[SteamSearchCandidate]] = None,
    *,
    total: int = 10,
    invalid_count: int = 0,
    truncated_count: int = 0,
) -> SteamSearchResult:
    if candidates is None:
        candidates = [_candidate()]
    return SteamSearchResult(
        total=total,
        candidates=candidates,
        invalid_count=invalid_count,
        truncated_count=truncated_count,
    )


# ── Fixture ───────────────────────────────────────────────────────────────


@pytest.fixture
def app_with_fake_searcher(monkeypatch: pytest.MonkeyPatch):
    """Build the real app with Steam enabled and a fake searcher."""

    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", TEST_TOKEN)
    monkeypatch.setenv("STEAM_RETRIEVAL_ENABLED", "true")

    from app.main import create_app

    fake = FakeSearchRetriever()
    with TestClient(create_app()) as client:
        app = client.app
        app.state.steam_search_retriever = fake
        yield client, fake


def _auth() -> Dict[str, str]:
    return {HEADER_NAME: TEST_TOKEN}


# ── Authentication ────────────────────────────────────────────────────────


class TestAuthentication:
    """The search route requires the internal token, exactly like the
    generation routes."""

    def test_missing_token_returns_401(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        client, _ = app_with_fake_searcher
        response = client.post(ENDPOINT, json={"query": "Portal 2"})
        assert response.status_code == 401
        assert response.json()["code"] == "unauthorized"

    def test_incorrect_token_returns_401(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        client, _ = app_with_fake_searcher
        response = client.post(
            ENDPOINT,
            json={"query": "Portal 2"},
            headers={HEADER_NAME: "wrong-token"},
        )
        assert response.status_code == 401
        assert response.json()["code"] == "unauthorized"

    def test_missing_token_makes_no_upstream_call(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        client, fake = app_with_fake_searcher
        client.post(ENDPOINT, json={"query": "Portal 2"})
        assert fake.calls == []

    def test_correct_token_passes_auth(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        client, fake = app_with_fake_searcher
        fake.next_result = _result()
        response = client.post(ENDPOINT, json={"query": "Portal 2"}, headers=_auth())
        assert response.status_code == 200
        assert fake.calls == ["Portal 2"]

    def test_missing_query_field_returns_422(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        client, fake = app_with_fake_searcher
        response = client.post(ENDPOINT, json={}, headers=_auth())
        assert response.status_code == 422
        assert fake.calls == []


# ── Feature switch ────────────────────────────────────────────────────────


class TestFeatureSwitch:
    """Steam retrieval stays off unless the operator enables it."""

    def test_disabled_switch_returns_503(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", TEST_TOKEN)
        monkeypatch.setenv("STEAM_RETRIEVAL_ENABLED", "false")

        from app.main import create_app

        with TestClient(create_app()) as client:
            response = client.post(
                ENDPOINT, json={"query": "Portal 2"}, headers=_auth()
            )
            assert response.status_code == 503
            assert response.json()["code"] == "steam_search_disabled"

    def test_disabled_switch_makes_no_upstream_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", TEST_TOKEN)
        monkeypatch.setenv("STEAM_RETRIEVAL_ENABLED", "false")

        from app.main import create_app

        fake = FakeSearchRetriever()
        with TestClient(create_app()) as client:
            client.app.state.steam_search_retriever = fake
            client.post(ENDPOINT, json={"query": "Portal 2"}, headers=_auth())
            assert fake.calls == []

    def test_steam_retrieval_is_disabled_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The switch must not default to on: the app is built with no
        # STEAM_RETRIEVAL_ENABLED in the environment.
        monkeypatch.delenv("STEAM_RETRIEVAL_ENABLED", raising=False)
        monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", TEST_TOKEN)

        from app.config import Settings

        assert Settings().steam_retrieval_enabled is False


# ── Successful search ─────────────────────────────────────────────────────


class TestSuccessfulSearch:
    """The documented response shape."""

    def test_response_carries_the_candidates(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        client, fake = app_with_fake_searcher
        fake.next_result = _result(
            [_candidate(620, "Portal 2"), _candidate(323180, "Portal 2 Soundtrack")],
            total=10,
        )
        response = client.post(ENDPOINT, json={"query": "Portal 2"}, headers=_auth())
        assert response.status_code == 200
        body = response.json()
        assert body["candidate_count"] == 2
        assert body["total"] == 10
        assert body["invalid_count"] == 0
        assert body["truncated_count"] == 0
        assert body["candidates"][0]["app_id"] == 620
        assert body["candidates"][0]["name"] == "Portal 2"
        assert body["candidates"][0]["type"] == "app"
        assert body["candidates"][0]["tiny_image"].startswith("https://")

    def test_empty_results_return_200_with_no_candidates(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        client, fake = app_with_fake_searcher
        fake.next_result = _result([], total=0)
        response = client.post(ENDPOINT, json={"query": "No Such Game"}, headers=_auth())
        assert response.status_code == 200
        body = response.json()
        assert body["candidates"] == []
        assert body["candidate_count"] == 0
        assert body["total"] == 0

    def test_optional_candidate_fields_are_null(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        client, fake = app_with_fake_searcher
        fake.next_result = _result(
            [_candidate(620, "Portal 2", app_type=None, tiny_image=None)]
        )
        response = client.post(ENDPOINT, json={"query": "Portal 2"}, headers=_auth())
        candidate = response.json()["candidates"][0]
        assert candidate["type"] is None
        assert candidate["tiny_image"] is None

    def test_the_endpoint_makes_exactly_one_upstream_call(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        client, fake = app_with_fake_searcher
        fake.next_result = _result(
            [_candidate(620, "Portal 2"), _candidate(323180, "Portal 2 Soundtrack")]
        )
        client.post(ENDPOINT, json={"query": "Portal 2"}, headers=_auth())
        # One search call, and no per-candidate appdetails call.
        assert len(fake.calls) == 1

    def test_the_endpoint_does_not_filter_by_product_type(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        # A demo and a soundtrack are returned unchanged: the search
        # endpoint does not know what a demo is.
        client, fake = app_with_fake_searcher
        fake.next_result = _result(
            [
                _candidate(3516070, "Grand Emprise 2: Portals Apart Demo"),
                _candidate(323180, "Portal 2 Soundtrack"),
            ]
        )
        response = client.post(ENDPOINT, json={"query": "Portal 2"}, headers=_auth())
        names = [c["name"] for c in response.json()["candidates"]]
        assert "Grand Emprise 2: Portals Apart Demo" in names
        assert "Portal 2 Soundtrack" in names


# ── Response hygiene ──────────────────────────────────────────────────────


class TestResponseHygiene:
    """The response exposes only what a caller needs to pick a candidate."""

    def test_response_excludes_raw_steam_fields(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        client, fake = app_with_fake_searcher
        fake.next_result = _result(
            [
                _candidate(
                    620,
                    "Portal 2",
                    price=SteamSearchPrice(currency="VND", initial=14000000.0, final=14000000.0),
                    metascore=95,
                )
            ]
        )
        response = client.post(ENDPOINT, json={"query": "Portal 2"}, headers=_auth())
        candidate = response.json()["candidates"][0]
        assert set(candidate.keys()) == {"app_id", "name", "type", "tiny_image"}
        assert "price" not in candidate
        assert "metascore" not in candidate
        assert "platforms" not in candidate
        assert "streamingvideo" not in candidate
        assert "controller_support" not in candidate

    def test_response_has_no_evidence_block(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        client, fake = app_with_fake_searcher
        fake.next_result = _result()
        response = client.post(ENDPOINT, json={"query": "Portal 2"}, headers=_auth())
        body = response.json()
        assert "evidence" not in body
        assert "conflicts" not in body
        assert "status" not in body

    def test_the_query_is_not_echoed_back(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        client, fake = app_with_fake_searcher
        fake.next_result = _result()
        response = client.post(
            ENDPOINT, json={"query": "Secret Portal Title"}, headers=_auth()
        )
        assert "Secret Portal Title" not in response.text


# ── Invalid query ─────────────────────────────────────────────────────────


class TestInvalidQuery:
    """The search core's validation is reflected as HTTP 422."""

    def test_empty_query_returns_422(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        client, fake = app_with_fake_searcher
        fake.next_error = SteamSearchQueryInvalidError(outcome="invalid_query")
        response = client.post(ENDPOINT, json={"query": "   "}, headers=_auth())
        assert response.status_code == 422
        assert response.json()["code"] == "invalid_steam_search_query"

    def test_over_long_query_returns_422(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        client, fake = app_with_fake_searcher
        fake.next_error = SteamSearchQueryInvalidError(outcome="invalid_query")
        response = client.post(
            ENDPOINT, json={"query": "x" * 500}, headers=_auth()
        )
        assert response.status_code == 422

    def test_an_invalid_query_is_not_echoed_in_the_error(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        client, fake = app_with_fake_searcher
        fake.next_error = SteamSearchQueryInvalidError(outcome="invalid_query")
        response = client.post(
            ENDPOINT, json={"query": "Secret Portal Title"}, headers=_auth()
        )
        assert "Secret Portal Title" not in response.text


# ── Cooldown ──────────────────────────────────────────────────────────────


class TestCooldown:
    """The repeated-query cooldown surfaces as HTTP 429."""

    def test_cooldown_returns_429(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        client, fake = app_with_fake_searcher
        fake.next_error = SteamSearchCooldownError(outcome="cooldown")
        response = client.post(ENDPOINT, json={"query": "Portal 2"}, headers=_auth())
        assert response.status_code == 429
        assert response.json()["code"] == "steam_search_cooldown"

    def test_cooldown_and_upstream_rate_limit_are_distinguishable(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        client, fake = app_with_fake_searcher
        fake.next_error = SteamRetrievalError(outcome="rate_limited")
        response = client.post(ENDPOINT, json={"query": "Portal 2"}, headers=_auth())
        assert response.status_code == 429
        assert response.json()["code"] == "steam_rate_limited"


# ── Upstream failures ─────────────────────────────────────────────────────


class TestUpstreamFailures:
    """An upstream failure is a controlled error, never a raw leak."""

    @pytest.mark.parametrize(
        "outcome",
        [
            "http_error",
            "timeout",
            "transport_error",
            "unexpected_content_type",
            "response_too_large",
            "invalid_json",
            "unexpected_structure",
        ],
    )
    def test_upstream_failures_return_502(
        self,
        outcome: str,
        app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever],
    ) -> None:
        client, fake = app_with_fake_searcher
        fake.next_error = SteamRetrievalError(outcome=outcome)
        response = client.post(ENDPOINT, json={"query": "Portal 2"}, headers=_auth())
        assert response.status_code == 502
        assert response.json()["code"] == "steam_search_unavailable"

    def test_upstream_rate_limit_returns_429(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        client, fake = app_with_fake_searcher
        fake.next_error = SteamRetrievalError(outcome="rate_limited")
        response = client.post(ENDPOINT, json={"query": "Portal 2"}, headers=_auth())
        assert response.status_code == 429
    def test_the_upstream_body_is_never_returned(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        client, fake = app_with_fake_searcher
        fake.next_error = SteamRetrievalError(
            "upstream said: {\"secret\": \"leak\"}", outcome="http_error"
        )
        response = client.post(ENDPOINT, json={"query": "Portal 2"}, headers=_auth())
        assert response.status_code == 502
        assert "leak" not in response.text
        assert "secret" not in response.text

    def test_the_internal_url_is_never_returned(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        client, fake = app_with_fake_searcher
        fake.next_error = SteamRetrievalError(
            "failed on https://store.steampowered.com/api/storesearch/",
            outcome="transport_error",
        )
        response = client.post(ENDPOINT, json={"query": "Portal 2"}, headers=_auth())
        assert "store.steampowered.com" not in response.text

    def test_the_token_is_never_returned(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        client, fake = app_with_fake_searcher
        fake.next_error = SteamRetrievalError(outcome="http_error")
        response = client.post(ENDPOINT, json={"query": "Portal 2"}, headers=_auth())
        assert TEST_TOKEN not in response.text


# ── The endpoint does not touch the evidence pipeline ────────────────────


class TestEvidenceIsolation:
    """A search result is a candidate, never evidence."""

    def test_the_search_route_does_not_register_evidence(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        # The evidence builder is never invoked by the search route: the
        # route does not import it, and the response carries no evidence.
        assert not _imported_modules_mentioning("evidence")

    def test_the_translation_service_is_not_called(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        client, fake = app_with_fake_searcher
        fake.next_result = _result()

        calls: List[Any] = []
        service = getattr(client.app.state, "translation_service", None)
        if service is not None:
            original = service.generate
            service.generate = lambda request: calls.append(request)  # type: ignore[assignment]

        client.post(ENDPOINT, json={"query": "Portal 2"}, headers=_auth())
        assert calls == []


def _route_module_source() -> str:
    """Return the source of the search route module."""

    from app.routers import steam_search as route_module

    import inspect

    return inspect.getsource(route_module)


def _imported_modules_mentioning(fragment: str) -> List[str]:
    """Return the modules the search route imports that mention a fragment.

    The route's *docstring* may talk about the evidence pipeline, but it
    must not **import** anything from it. The check therefore works on
    the parsed import statements, not on the raw source text.
    """

    import ast

    tree = ast.parse(_route_module_source())
    imported: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
            imported.extend(
                f"{node.module}.{alias.name}" for alias in node.names
            )
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    return [name for name in imported if fragment in name]


# ── The existing endpoints are unchanged ──────────────────────────────────


class TestExistingEndpointsUnchanged:
    """The translation contract is untouched."""

    def test_the_translation_endpoint_still_exists(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        client, _ = app_with_fake_searcher
        response = client.post(
            "/internal/descriptions/generate",
            json={"product_name": "Aurora's Echo", "steam_app_id": 620},
            headers=_auth(),
        )
        assert response.status_code == 503

    def test_the_removed_legacy_endpoint_is_gone(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        client, _ = app_with_fake_searcher
        response = client.post(
            "/internal/descriptions/generate",
            json={"product_name": "Aurora's Echo", "developer": "Studio"},
            headers=_auth(),
        )
        # The earlier route was unmounted together with its Gemini-backed
        # pipeline. The path is answered again — by the translation
        # route, which was renamed onto it — but the old *contract* is
        # gone: that payload carries a developer and no App ID, and the
        # translation contract rejects it outright.
        assert response.status_code == 422

    def test_the_removed_structured_draft_endpoint_is_gone(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        client, _ = app_with_fake_searcher
        response = client.post(
            "/internal/descriptions/generate-v2",
            json={"product_name": "Aurora's Echo", "developer": "Studio"},
            headers=_auth(),
        )
        # The structured-draft route was unmounted together with its
        # pipeline, so it no longer answers at all.
        assert response.status_code == 404

    def test_the_health_endpoint_is_unchanged(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        client, _ = app_with_fake_searcher
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["service"] == "hotwatergas-ai"

    def test_the_openapi_contract_declares_the_route(
        self, app_with_fake_searcher: tuple[TestClient, FakeSearchRetriever]
    ) -> None:
        client, _ = app_with_fake_searcher
        schema = client.get("/openapi.json").json()
        assert "/internal/steam/search" in schema["paths"]
        assert "post" in schema["paths"]["/internal/steam/search"]


# ── Integration: the real search core behind the route ───────────────────


class TestRealSearchCoreIntegration:
    """The route wired to the **real** ``SteamSearchRetriever``.

    The retriever's transport is mocked with ``httpx.MockTransport`` and
    the captured fixture is served from disk, so the whole seam — route,
    core, mapper, fixture — is exercised without a network.
    """

    @pytest.fixture
    def client_with_real_core(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
        import httpx

        from app.main import create_app
        from app.retrieval.steam_search import SteamSearchRetriever

        monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", TEST_TOKEN)
        monkeypatch.setenv("STEAM_RETRIEVAL_ENABLED", "true")

        payload = json.loads(
            (FIXTURES / "storesearch_portal_2.json").read_text(encoding="utf-8")
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=json.dumps(payload).encode("utf-8"),
                headers={"content-type": "application/json; charset=utf-8"},
            )

        transport = httpx.MockTransport(handler)
        client_http = httpx.Client(transport=transport)
        retriever = SteamSearchRetriever(
            connect_timeout_seconds=1.0,
            read_timeout_seconds=1.0,
            cooldown_seconds=2.0,
            client=client_http,
        )

        with TestClient(create_app()) as client:
            client.app.state.steam_search_retriever = retriever
            yield client

    def test_the_fixture_is_returned_through_the_route(
        self, client_with_real_core: TestClient
    ) -> None:
        response = client_with_real_core.post(
            ENDPOINT, json={"query": "Portal 2"}, headers=_auth()
        )
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 10
        assert body["candidate_count"] == 10
        assert body["candidates"][0]["app_id"] == 620
        assert body["candidates"][0]["name"] == "Portal 2"

    def test_the_route_response_matches_the_candidate_fields(
        self, client_with_real_core: TestClient
    ) -> None:
        response = client_with_real_core.post(
            ENDPOINT, json={"query": "Portal 2"}, headers=_auth()
        )
        for candidate in response.json()["candidates"]:
            assert set(candidate.keys()) == {"app_id", "name", "type", "tiny_image"}

    def test_the_real_cooldown_surfaces_as_429(
        self, client_with_real_core: TestClient
    ) -> None:
        first = client_with_real_core.post(
            ENDPOINT, json={"query": "Portal 2"}, headers=_auth()
        )
        assert first.status_code == 200
        second = client_with_real_core.post(
            ENDPOINT, json={"query": "Portal 2"}, headers=_auth()
        )
        assert second.status_code == 429
        assert second.json()["code"] == "steam_search_cooldown"

    def test_a_real_invalid_query_surfaces_as_422(
        self, client_with_real_core: TestClient
    ) -> None:
        response = client_with_real_core.post(
            ENDPOINT, json={"query": "   "}, headers=_auth()
        )
        assert response.status_code == 422
        assert response.json()["code"] == "invalid_steam_search_query"

    def test_a_real_upstream_rate_limit_surfaces_as_429(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from app.main import create_app
        from app.retrieval.steam_search import SteamSearchRetriever

        monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", TEST_TOKEN)
        monkeypatch.setenv("STEAM_RETRIEVAL_ENABLED", "true")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                content=b"{}",
                headers={"content-type": "application/json"},
            )

        retriever = SteamSearchRetriever(
            connect_timeout_seconds=1.0,
            read_timeout_seconds=1.0,
            cooldown_seconds=0.0,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        with TestClient(create_app()) as client:
            client.app.state.steam_search_retriever = retriever
            response = client.post(
                ENDPOINT, json={"query": "Portal 2"}, headers=_auth()
            )
        assert response.status_code == 429
        assert response.json()["code"] == "steam_rate_limited"

    def test_a_real_upstream_failure_surfaces_as_502(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from app.main import create_app
        from app.retrieval.steam_search import SteamSearchRetriever

        monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", TEST_TOKEN)
        monkeypatch.setenv("STEAM_RETRIEVAL_ENABLED", "true")

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        retriever = SteamSearchRetriever(
            connect_timeout_seconds=1.0,
            read_timeout_seconds=1.0,
            cooldown_seconds=0.0,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        with TestClient(create_app()) as client:
            client.app.state.steam_search_retriever = retriever
            response = client.post(
                ENDPOINT, json={"query": "Portal 2"}, headers=_auth()
            )
        assert response.status_code == 502
        assert response.json()["code"] == "steam_search_unavailable"

