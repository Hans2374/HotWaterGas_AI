"""Self-checks for the offline-safety guards.

The guards added to ``tests/conftest.py`` are only worth something if
they actually work. These tests prove that:

* a real HTTP request is **blocked** by the egress guard,
* a mocked HTTP request is **not** blocked by it,
* a real DNS resolution of a non-loopback host is **blocked**,
* the ``.env`` file source is **not** read when ``Settings`` is built.

If a future change to the guards, to ``httpx``, or to
``Settings.model_config`` silently disables one of them, these tests
fail. A guard that quietly stops working is worse than no guard at
all, because it looks like a safety net.
"""

from __future__ import annotations

import socket

import httpx
import pytest

from app.config import Settings


class TestNetworkEgressGuard:
    """The egress guard blocks real requests and spares mocked ones."""

    def test_a_real_http_request_is_blocked(self) -> None:
        with pytest.raises(AssertionError) as excinfo:
            httpx.get("http://127.0.0.1:11434/api/tags", timeout=1.0)
        assert "Network access is blocked" in str(excinfo.value)

    def test_a_mocked_http_request_is_not_blocked(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": True})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        assert client.get("http://example.invalid/x").json() == {"ok": True}

    def test_a_real_dns_resolution_is_blocked(self) -> None:
        with pytest.raises(AssertionError) as excinfo:
            socket.getaddrinfo("example.invalid", 443)
        assert "Network access is blocked" in str(excinfo.value)

    def test_loopback_resolution_is_still_allowed(self) -> None:
        # Loopback resolution must keep working: the test client and
        # the mocked transports rely on it.
        results = socket.getaddrinfo("127.0.0.1", 80)
        assert results


class TestDotenvGuard:
    """``Settings`` must not read a ``.env`` file in the test environment."""

    def test_the_settings_class_still_declares_the_dotenv_source(self) -> None:
        # The application needs the dotenv source outside the test
        # suite. The guard removes it *during* the suite, so the class
        # declaration must remain.
        assert Settings.model_config.get("env_file") == ".env"

    def test_settings_can_be_built_without_reading_a_file(self) -> None:
        settings = Settings()
        assert settings.ai_generation_enabled is False
        assert settings.internal_service_token == ""

    def test_environment_variables_are_still_honoured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The guard removes the *file* source only. The environment
        # source must keep working, otherwise the test suite could not
        # configure the service at all.
        monkeypatch.setenv("AI_GENERATION_ENABLED", "true")
        assert Settings().ai_generation_enabled is True


class TestSteamSwitchIsIsolated:
    """``STEAM_RETRIEVAL_ENABLED`` must not leak between tests.

    It was missing from the shared list of settings environment keys,
    so a test that enabled it would have leaked the value into every
    later test that did not clear it.
    """

    def test_the_switch_is_in_the_cleared_key_list(self) -> None:
        from tests.conftest import _SETTINGS_ENV_KEYS

        assert "STEAM_RETRIEVAL_ENABLED" in _SETTINGS_ENV_KEYS

    def test_the_switch_is_cleared_by_the_clean_env_fixture(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import os

        # ``clean_env`` already ran when the test body started, so the
        # value is absent at this point.
        assert os.environ.get("STEAM_RETRIEVAL_ENABLED") is None
        assert Settings().steam_retrieval_enabled is False

    def test_the_set_env_helper_can_configure_the_switch(
        self, set_env: "callable"
    ) -> None:
        import os

        # Before this key was added to the shared list, the helper
        # raised ``KeyError`` for it, so no test could configure the
        # Steam switch through the supported path.
        set_env(STEAM_RETRIEVAL_ENABLED="true")
        assert os.environ.get("STEAM_RETRIEVAL_ENABLED") == "true"
        assert Settings().steam_retrieval_enabled is True
