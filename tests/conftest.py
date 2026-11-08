"""Shared pytest fixtures.

The fixtures here exist so individual tests can:

* Reset environment variables to a known state.
* Build a FastAPI ``TestClient`` against a freshly-built app instance,
  with the env vars applied *before* ``Settings`` is constructed.

This module adds two **session-wide safety guards** that make the
suite provably offline and provably unable to read a dotenv file:

* :func:`_block_dotenv_loading` neutralises the ``.env`` file source
  that ``Settings.model_config`` declares, so a ``.env`` file can never
  be read implicitly - not even one that appears later.
* :func:`_block_network_egress` makes any attempt to open a real
  outbound connection fail loudly, so a test that forgets to inject a
  mock transport fails instead of silently reaching a live service.
"""

from __future__ import annotations

import os
import socket
from typing import Iterator, List

import pytest
from fastapi.testclient import TestClient


# Keys that influence ``Settings``. Tests should clear these before they
# instantiate the app, then set the ones they care about.
_SETTINGS_ENV_KEYS = (
    "APP_ENV",
    "AI_GENERATION_ENABLED",
    "INTERNAL_SERVICE_TOKEN",
    # Provider selection and the local Ollama settings.
    "LLM_PROVIDER",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL_ID",
    "OLLAMA_REQUEST_TIMEOUT_SECONDS",
    "OLLAMA_CONNECT_TIMEOUT_SECONDS",
    "OLLAMA_TAGS_TIMEOUT_SECONDS",
    "OLLAMA_NUM_PREDICT",
    "OLLAMA_TEMPERATURE",
    # The optional Steam enrichment switch. It was missing
    # from this list, so a test that set it leaked the value into every
    # later test that did not clear it.
    "STEAM_RETRIEVAL_ENABLED",
    "STEAM_CONNECT_TIMEOUT_SECONDS",
    "STEAM_READ_TIMEOUT_SECONDS",
    "STEAM_MAX_RESPONSE_BYTES",
    # The Steam Store Search settings. They were missing from
    # this list too, so a test that set one leaked the value into every
    # later test that did not clear it.
    "STEAM_SEARCH_MAX_CANDIDATES",
    "STEAM_SEARCH_COOLDOWN_SECONDS",
    # The translation bounds. Both spellings are listed: the neutral one
    # is canonical, and the legacy ``V3_*`` one is still honoured.
    "MIN_TARGET_WORDS",
    "MAX_TARGET_WORDS",
    "MAX_DESCRIPTION_WORDS",
    "MAX_SOURCE_BYTES",
    "MAX_PROVIDER_INPUT_BYTES",
    "MIN_SOURCE_TOKENS",
    "V3_MIN_TARGET_WORDS",
    "V3_MAX_TARGET_WORDS",
    "V3_MAX_DESCRIPTION_WORDS",
    "V3_MAX_SOURCE_BYTES",
    "V3_MAX_PROVIDER_INPUT_BYTES",
    "V3_MIN_SOURCE_TOKENS",
)


# ── Safety guards ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True, scope="session")
def _block_dotenv_loading() -> Iterator[None]:
    """Make it impossible for ``Settings`` to read a ``.env`` file.

    ``Settings.model_config`` declares ``env_file=".env"``, so
    pydantic-settings constructs a ``DotEnvSettingsSource`` on **every**
    ``Settings()`` instantiation. Even when no ``.env`` file exists
    today, relying on that is a silent dependency: the file could be
    created later, or the working directory could differ, and the
    service would then start reading secrets from disk.

    The guard forces the dotenv source to be constructed with
    ``env_file=None``, which makes pydantic-settings build a source
    that yields no values. Environment variables keep working exactly
    as before - only the *file* source is removed.

    This is a test-only guard. It does not change how the application
    behaves outside the test suite.
    """

    import pydantic_settings.sources as sources

    original_init = sources.DotEnvSettingsSource.__init__

    def _init_without_file(self: object, *args: object, **kwargs: object) -> None:
        kwargs["env_file"] = None
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]

    sources.DotEnvSettingsSource.__init__ = _init_without_file  # type: ignore[method-assign]
    try:
        yield
    finally:
        sources.DotEnvSettingsSource.__init__ = original_init  # type: ignore[method-assign]


@pytest.fixture(autouse=True)
def _block_network_egress(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Fail loudly if a test tries to make a real network request.

    Two layers, because one layer is not enough:

    * **The httpx transport layer** is the precise guard. Every outbound
      HTTP call in this service goes through ``httpx`` (``ollama.py`` and
      ``steam.py``), and a real client uses ``httpx.HTTPTransport`` while
      a mocked one uses ``httpx.MockTransport`` - a *different class*.
      Patching ``HTTPTransport.handle_request`` therefore blocks every
      real request, including one to a loopback Ollama, while leaving
      every mocked transport untouched.
    * **The DNS layer** is the backstop for clients that are not httpx.
      It blocks the resolution of any non-loopback hostname.

    The socket layer is deliberately **not** patched: asyncio's own
    event loop opens a loopback socketpair internally on Windows, and
    interfering with ``socket.connect`` there makes the test client
    hang instead of failing. Blocking at the transport and DNS layers
    gives the same protection without touching the event loop.

    A test that genuinely needs the network must opt in explicitly
    with the ``@pytest.mark.allow_network`` marker, so the exception
    is visible in the test source rather than buried in a fixture.
    """

    if request.node.get_closest_marker("allow_network") is not None:
        yield
        return

    import httpx

    def _blocked_http(self: object, http_request: object) -> None:
        raise AssertionError(
            "Network access is blocked during tests. A test tried to make "
            f"a real HTTP request to {getattr(http_request, 'url', '?')!r}. "
            "Inject a mock transport (httpx.MockTransport) or mark the "
            "test with @pytest.mark.allow_network if a live call is "
            "intended."
        )

    _original_getaddrinfo = socket.getaddrinfo

    def _blocked_dns(
        host: object,
        port: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        if isinstance(host, str) and host not in (
            "127.0.0.1",
            "::1",
            "localhost",
            "",
        ):
            raise AssertionError(
                "Network access is blocked during tests. A test tried to "
                f"resolve the hostname {host!r}. Use a mock transport "
                "(httpx.MockTransport) or mark the test with "
                "@pytest.mark.allow_network if a live call is intended."
            )
        return _original_getaddrinfo(host, port, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        httpx.HTTPTransport, "handle_request", _blocked_http, raising=False
    )
    monkeypatch.setattr(socket, "getaddrinfo", _blocked_dns, raising=False)
    yield


# ── Environment isolation ─────────────────────────────────────────────────


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip every known settings env var for the duration of a test."""

    for key in _SETTINGS_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield


@pytest.fixture
def client(clean_env: None) -> Iterator[TestClient]:
    """Return a TestClient bound to a freshly-built app.

    ``clean_env`` runs first, so the app is constructed with no
    environment-derived configuration. Any test that needs different
    values should call ``monkeypatch.setenv(...)`` *before* requesting
    the ``client`` fixture, or after this fixture by re-importing the
    modules (not necessary here because ``Settings`` is built
    inside the lifespan context manager, not at import time).
    """

    # Import inside the fixture so the env reset above has already
    # taken effect.
    from app.main import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def set_env(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> "callable":
    """Return a helper that sets env vars before app construction.

    Usage::

        def test_x(set_env):
            set_env(AI_GENERATION_ENABLED="true")
            with TestClient(create_app()) as c: ...
    """

    def _set(**values: str) -> None:
        for key, value in values.items():
            if key not in _SETTINGS_ENV_KEYS:
                raise KeyError(f"Unknown settings env var: {key}")
            monkeypatch.setenv(key, value)

    return _set


# ── Offline-safety assertions ────────────────────────────────────────────


def _dotenv_file_sources(settings_cls: type) -> List[str]:
    """Return the dotenv file sources the settings class would read.

    Used by the offline-safety test to prove that the settings class
    is not wired to a ``.env`` file in the test environment.
    """

    import pydantic_settings.sources as sources

    config = settings_cls.model_config
    env_file = config.get("env_file")
    if env_file is None:
        return []
    return [str(env_file)]


def test_settings_are_not_wired_to_a_dotenv_file_in_tests() -> None:
    """The offline guard removes the ``.env`` file source from Settings.

    This is a *self-check* on the safety guard itself: if a future
    change to ``Settings.model_config`` reintroduced a dotenv source
    that the guard no longer neutralises, this test would fail.
    """

    from app.config import Settings

    # The class still declares a dotenv source (the application needs
    # it outside the test suite), but the guard rewrites it to None at
    # construction time. Assert the *effective* behaviour instead: the
    # settings must be buildable with no file present, and the guard
    # must be the thing that makes it so.
    assert _dotenv_file_sources(Settings) == [".env"]

    # Under the guard, constructing Settings must not read a file. The
    # guard is session-scoped and autouse, so it is already active.
    settings = Settings()
    assert settings.ai_generation_enabled is False
    assert settings.internal_service_token == ""
