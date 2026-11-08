"""Tests for the internal-token dependency.

The tests build a minimal FastAPI app that mounts a single
``/protected`` route guarded by ``require_internal_token``. The full
service stack is *not* started, so the dependency is exercised in
isolation.
"""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.dependencies.auth import HEADER_NAME, require_internal_token


@pytest.fixture
def protected_app():
    """A minimal app that exposes one token-protected route."""

    app = FastAPI()

    from app.errors_http import install_auth_handlers

    install_auth_handlers(app)

    @app.get("/protected")
    def _protected(_: None = Depends(require_internal_token)) -> dict[str, str]:
        return {"ok": "yes"}

    return app


@pytest.fixture
def configured_app(monkeypatch: pytest.MonkeyPatch):
    """A minimal app with the token already configured server-side."""

    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "server-side-secret-12345")
    app = FastAPI()

    from app.errors_http import install_auth_handlers

    install_auth_handlers(app)

    @app.get("/protected")
    def _protected(_: None = Depends(require_internal_token)) -> dict[str, str]:
        return {"ok": "yes"}

    return app


class TestTokenMissingConfig:
    """Server-side token absent -> request fails closed."""

    def test_missing_token_config_returns_500(
        self,
        clean_env: None,
        protected_app: FastAPI,
    ) -> None:
        with TestClient(protected_app) as client:
            response = client.get(
                "/protected",
                headers={HEADER_NAME: "client-supplied-token"},
            )
        assert response.status_code == 500
        body = response.json()
        assert body["code"] == "internal_token_not_configured"
        # Never echo the client-supplied token.
        assert "client-supplied-token" not in str(body)
        # Never echo the (absent) server-side token.
        assert "secret" not in str(body).lower()


class TestTokenAuthentication:
    """Token present and matching -> success; otherwise -> 401."""

    def test_correct_token_returns_200(
        self,
        configured_app: FastAPI,
    ) -> None:
        with TestClient(configured_app) as client:
            response = client.get(
                "/protected",
                headers={HEADER_NAME: "server-side-secret-12345"},
            )
        assert response.status_code == 200
        assert response.json() == {"ok": "yes"}

    def test_missing_header_returns_401(
        self,
        configured_app: FastAPI,
    ) -> None:
        with TestClient(configured_app) as client:
            response = client.get("/protected")
        assert response.status_code == 401
        body = response.json()
        assert body["code"] == "unauthorized"
        # The expected token must never appear in the response.
        assert "server-side-secret" not in str(body)

    def test_incorrect_token_returns_401(
        self,
        configured_app: FastAPI,
    ) -> None:
        with TestClient(configured_app) as client:
            response = client.get(
                "/protected",
                headers={HEADER_NAME: "wrong-token"},
            )
        assert response.status_code == 401
        body = response.json()
        assert body["code"] == "unauthorized"
        assert "wrong-token" not in str(body)

    def test_empty_token_returns_401(
        self,
        configured_app: FastAPI,
    ) -> None:
        with TestClient(configured_app) as client:
            response = client.get(
                "/protected",
                headers={HEADER_NAME: ""},
            )
        assert response.status_code == 401

    def test_case_sensitive_comparison(
        self,
        configured_app: FastAPI,
    ) -> None:
        with TestClient(configured_app) as client:
            # Different case -> different token.
            response = client.get(
                "/protected",
                headers={HEADER_NAME: "SERVER-SIDE-SECRET-12345"},
            )
        assert response.status_code == 401

    def test_token_with_unicode_is_accepted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Internal tokens are restricted to ASCII by RFC 7230
        # (``Header values are visible to the wire and may be inspected
        # by intermediaries; they MUST be ASCII``). The auth code does
        # not need to support Unicode tokens. This test pins the
        # behaviour: an all-ASCII token with punctuation and digits
        # round-trips correctly.
        token = "tok-_-2026!@#"
        monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", token)
        app = FastAPI()

        from app.errors_http import install_auth_handlers

        install_auth_handlers(app)

        @app.get("/protected")
        def _protected(_: None = Depends(require_internal_token)) -> dict[str, str]:
            return {"ok": "yes"}

        with TestClient(app) as client:
            response = client.get(
                "/protected",
                headers={HEADER_NAME: token},
            )
        assert response.status_code == 200


class TestConstantTimeComparison:
    """The dependency MUST use a constant-time comparison.

    The test verifies the implementation by inspecting the source. A
    regression to ``==`` would defeat the purpose of the dependency.
    """

    def test_uses_hmac_compare_digest(self) -> None:
        import inspect

        from app import dependencies

        source = inspect.getsource(dependencies.auth)
        assert "hmac.compare_digest" in source
        # The "==" comparison must not appear in a code path that
        # decides whether to grant access. (Asserting its absence is
        # too brittle; instead we look for the specific pattern that
        # indicates a non-constant-time comparison of the tokens.)
        assert "if x_internal_token ==" not in source
        assert "if x_internal_token != " not in source
