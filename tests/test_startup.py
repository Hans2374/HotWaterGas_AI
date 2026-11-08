"""Startup tests.

The service must start with no provider
credential when generation is disabled. These tests verify that:

* ``create_app()`` returns a usable FastAPI app in a clean environment.
* The lifespan context manager runs without raising.
* A TestClient can be opened and ``GET /health`` succeeds.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import create_app


class TestAppStartup:
    """The app boots without a provider credential when generation is disabled."""

    def test_create_app_returns_fastapi_instance(self, clean_env: None) -> None:
        app = create_app()
        assert isinstance(app, FastAPI)

    def test_app_starts_without_a_provider_credential(self, clean_env: None) -> None:
        # Entering the TestClient context manager runs the lifespan
        # startup handler, which is where Settings are resolved.
        app = create_app()
        with TestClient(app) as client:
            response = client.get("/health")
        assert response.status_code == 200

    def test_settings_are_attached_to_app_state(self, clean_env: None) -> None:
        from app.config import Settings

        app = create_app()
        with TestClient(app) as client:
            # The lifespan handler stores the resolved Settings on
            # ``app.state`` so routers can read it without re-parsing
            # environment variables per request.
            assert isinstance(app.state.settings, Settings)
            assert app.state.settings.ai_generation_enabled is False
            # Also reachable via the live request cycle.
            response = client.get("/health")
        assert response.status_code == 200

    def test_openapi_schema_exposes_health_route(self, clean_env: None) -> None:
        app = create_app()
        with TestClient(app) as client:
            response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "/health" in schema["paths"]
        assert "get" in schema["paths"]["/health"]
