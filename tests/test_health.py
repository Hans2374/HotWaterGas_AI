"""Health-endpoint tests.

These cover the documented health-route contract:

* ``GET /health`` returns HTTP 200.
* The response JSON has the documented fields.
* The endpoint works when ``AI_GENERATION_ENABLED`` is ``false`` and no
  provider is configured.
* The endpoint does not require a provider credential to succeed.
* The endpoint is safe to call without any environment variables.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


EXPECTED_KEYS = {"status", "service", "version", "ai_generation_enabled"}


class TestHealthEndpoint:
    """Liveness contract for the health endpoint."""

    def test_returns_200_with_expected_payload(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

        body = response.json()
        assert set(body.keys()) == EXPECTED_KEYS
        assert body["status"] == "ok"
        assert body["service"] == "hotwatergas-ai"
        assert isinstance(body["version"], str)
        assert body["version"]  # Non-empty
        assert body["ai_generation_enabled"] is False

    def test_does_not_require_a_provider_credential(self, client: TestClient) -> None:
        # The default fixture strips every provider env var, so this is
        # implicit, but we state it explicitly for future readers.
        response = client.get("/health")
        assert response.status_code == 200

    def test_does_not_leak_secrets(self, client: TestClient) -> None:
        # The body must not include any configuration values that could
        # be sensitive (api key, model id, internal token, env, etc.).
        response = client.get("/health")
        body = response.json()
        for forbidden in ("apiKey", "api_key", "token", "secret", "key", "model"):
            assert forbidden not in body

    def test_reports_generation_flag_truthy_when_enabled(
        self,
        set_env,
        client: TestClient,  # noqa: ARG002 - documented dependency order
    ) -> None:
        # Note: ``client`` is requested after ``set_env`` to ensure the
        # env var is set before the app is built. The ``set_env`` fixture
        # itself depends on ``clean_env``.
        set_env(AI_GENERATION_ENABLED="true")
        from app.main import create_app

        with TestClient(create_app()) as app_client:
            response = app_client.get("/health")
        assert response.status_code == 200
        assert response.json()["ai_generation_enabled"] is True
