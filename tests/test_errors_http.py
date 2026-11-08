"""Unit tests for the shared, version-neutral HTTP error module.

The module under test is :mod:`app.errors_http`. It owns two things
that are part of the observable API contract:

* the error envelope (``{"code": ..., "message": ...}``) and the canned
  public wording of every documented error, and
* the exception handlers that turn a domain exception into a sanitised
  response.

The most important property is the **sanitisation**: an exception's own
message may embed a credential, a file path or a raw provider response,
so it must never reach the client. The tests below assert that the
original message is dropped and that only the canned wording is
returned.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.errors import (
    GenerationDisabledError,
    HotWaterGasAIError,
    InvalidInternalTokenError,
    InvalidProviderOutputError,
    MissingInternalTokenConfigError,
    ProviderTimeoutError,
    RequestTooLargeError,
    SteamIdentityMismatchError,
    SteamNotFoundError,
    SteamSearchCooldownError,
)
from app.errors_http import (
    _error_payload,
    _error_response,
    _GENERIC_MESSAGES,
    _safe_message,
    install_auth_handlers,
    register_handlers,
)


# ── The envelope ──────────────────────────────────────────────────────────


class TestErrorPayload:
    def test_the_envelope_has_exactly_two_keys(self) -> None:
        payload = _error_payload("some_code", "some message")
        assert set(payload) == {"code", "message"}

    def test_the_envelope_keeps_the_values(self) -> None:
        payload = _error_payload("some_code", "some message")
        assert payload["code"] == "some_code"
        assert payload["message"] == "some message"

    def test_the_envelope_is_json_serialisable(self) -> None:
        payload = _error_payload("some_code", "some message")
        assert json.loads(json.dumps(payload)) == payload


# ── The canned messages ───────────────────────────────────────────────────


class TestSafeMessage:
    def test_a_known_code_returns_its_canned_message(self) -> None:
        assert _safe_message("generation_disabled") == _GENERIC_MESSAGES[
            "generation_disabled"
        ]

    def test_an_unknown_code_returns_a_generic_message(self) -> None:
        message = _safe_message("a_code_nobody_documented")
        assert message
        # The fallback must not be the code itself, and must not leak
        # anything the caller supplied.
        assert "a_code_nobody_documented" not in message

    def test_every_documented_code_has_a_non_empty_message(self) -> None:
        for code in _GENERIC_MESSAGES:
            assert _safe_message(code).strip()
            assert _safe_message(code) != code

    def test_no_canned_message_contains_a_credential_like_token(self) -> None:
        # The canned wording is public, so it must not contain anything
        # that looks like a secret.
        for code, message in _GENERIC_MESSAGES.items():
            lowered = message.lower()
            assert "api_key" not in lowered, code
            assert "api key" not in lowered, code
            assert "password" not in lowered, code
            assert "secret" not in lowered, code
            assert "bearer" not in lowered, code


# ── Mapping a domain exception to a response ──────────────────────────────


class TestErrorResponse:
    def test_the_status_code_comes_from_the_exception(self) -> None:
        response = _error_response(ProviderTimeoutError())
        assert response.status_code == 504

    def test_the_public_code_comes_from_the_exception(self) -> None:
        response = _error_response(ProviderTimeoutError())
        assert json.loads(response.body)["code"] == "provider_timeout"

    def test_the_original_message_is_never_returned(self) -> None:
        secret = "sk-live-DO-NOT-LEAK /etc/secrets/token.txt"
        response = _error_response(HotWaterGasAIError(secret))
        body = json.loads(response.body)
        assert secret not in body["message"]
        assert secret not in response.body.decode("utf-8")

    def test_the_body_has_the_canonical_envelope(self) -> None:
        response = _error_response(GenerationDisabledError())
        body = json.loads(response.body)
        assert set(body) == {"code", "message"}
        assert body["code"] == "generation_disabled"
        assert body["message"] == _GENERIC_MESSAGES["generation_disabled"]

    def test_the_content_type_is_json(self) -> None:
        response = _error_response(GenerationDisabledError())
        assert response.media_type == "application/json"


# ── The handler bundle ────────────────────────────────────────────────────


def _app_with_bundle() -> FastAPI:
    app = FastAPI()
    register_handlers(app)

    @app.get("/boom")
    def _boom() -> None:
        raise SteamNotFoundError("internal detail that must not leak")

    @app.get("/mismatch")
    def _mismatch() -> None:
        raise SteamIdentityMismatchError("internal detail")

    @app.get("/too-large")
    def _too_large() -> None:
        raise RequestTooLargeError("internal detail")

    @app.get("/http-exception")
    def _http_exception() -> None:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "insufficient_product_data",
                "message": "a message the handler must replace",
            },
        )

    return app


class TestHandlerBundle:
    @pytest.fixture
    def client(self) -> TestClient:
        return TestClient(_app_with_bundle(), raise_server_exceptions=False)

    def test_a_domain_error_is_mapped_to_its_status(self, client: TestClient) -> None:
        response = client.get("/boom")
        assert response.status_code == 404
        assert response.json()["code"] == "steam_not_found"

    def test_a_domain_error_message_does_not_leak(self, client: TestClient) -> None:
        response = client.get("/boom")
        assert "internal detail" not in response.text

    def test_the_identity_mismatch_status_is_preserved(
        self, client: TestClient
    ) -> None:
        response = client.get("/mismatch")
        assert response.status_code == 409
        assert response.json()["code"] == "steam_identity_mismatch"

    def test_the_request_too_large_status_is_preserved(
        self, client: TestClient
    ) -> None:
        response = client.get("/too-large")
        assert response.status_code == 413
        assert response.json()["code"] == "request_too_large"

    def test_an_http_exception_is_reshaped_into_the_envelope(
        self, client: TestClient
    ) -> None:
        response = client.get("/http-exception")
        assert response.status_code == 422
        body = response.json()
        assert set(body) == {"code", "message"}
        assert body["code"] == "insufficient_product_data"
        # The canned wording replaces the one the route supplied.
        assert body["message"] == _GENERIC_MESSAGES["insufficient_product_data"]
        assert "a message the handler must replace" not in response.text

    def test_the_bundle_is_registered_exactly_once_per_exception_class(
        self, client: TestClient
    ) -> None:
        # Registering the bundle twice must not change the behaviour:
        # FastAPI keeps the most recent handler for a class, and the
        # bundle is idempotent.
        app = _app_with_bundle()
        register_handlers(app)
        doubled = TestClient(app, raise_server_exceptions=False)
        response = doubled.get("/boom")
        assert response.status_code == 404
        assert response.json()["code"] == "steam_not_found"


# ── The auth-only bundle ──────────────────────────────────────────────────


class TestInstallAuthHandlers:
    def test_the_missing_config_error_is_mapped(self) -> None:
        app = FastAPI()
        install_auth_handlers(app)

        @app.get("/protected")
        def _protected() -> None:
            raise MissingInternalTokenConfigError("internal detail")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/protected")
        assert response.status_code == 500
        assert response.json()["code"] == "internal_token_not_configured"
        assert "internal detail" not in response.text

    def test_the_invalid_token_error_is_mapped(self) -> None:
        app = FastAPI()
        install_auth_handlers(app)

        @app.get("/protected")
        def _protected() -> None:
            raise InvalidInternalTokenError("internal detail")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/protected")
        assert response.status_code == 401
        assert response.json()["code"] == "unauthorized"
        assert "internal detail" not in response.text


# ── The re-export removal ─────────────────────────────────────────────────


class TestRemovedRouterReExport:
    def test_the_removed_router_modules_are_gone(self) -> None:
        # The earlier description routers were removed together with
        # their pipelines. The module that now owns the description route
        # is the *retained* translation router, and it must not keep a
        # compatibility shim pointing at a deleted module.
        import importlib.util

        assert importlib.util.find_spec("app.routers.descriptions_v1") is None
        assert importlib.util.find_spec("app.routers.descriptions_v2") is None
        assert importlib.util.find_spec("app.routers.descriptions_v3") is None

    def test_the_helpers_live_in_the_neutral_module(self) -> None:
        # Every retained router imports the helpers from here.
        from app.routers import descriptions, steam_search

        assert descriptions._error_response is _error_response
        assert steam_search._error_payload is _error_payload
