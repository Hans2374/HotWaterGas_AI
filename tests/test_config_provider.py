"""Tests for the provider-selection configuration.

The tests pin the safety guarantees of the new settings:

* ``AI_GENERATION_ENABLED`` still defaults to ``False``.
* ``LLM_PROVIDER`` defaults to the local, free ``ollama`` provider.
* The Ollama defaults point at loopback and at ``qwen3:8b``.
* ``__repr__`` never leaks a secret and never leaks a credential that
  could be pasted elsewhere.
* An invalid provider name is rejected.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL_ID,
    DEFAULT_OLLAMA_REQUEST_TIMEOUT_SECONDS,
    Settings,
)


class TestOllamaDefaults:
    """The local development defaults."""

    def test_provider_defaults_to_ollama(self, clean_env: None) -> None:
        settings = Settings()
        assert settings.llm_provider == "ollama"

    def test_base_url_defaults_to_loopback(self, clean_env: None) -> None:
        settings = Settings()
        assert settings.ollama_base_url == DEFAULT_OLLAMA_BASE_URL
        assert settings.ollama_base_url == "http://127.0.0.1:11434"

    def test_base_url_is_loopback_only(self, clean_env: None) -> None:
        # The service must never point at a public Ollama endpoint.
        assert "127.0.0.1" in Settings().ollama_base_url
        assert "localhost" not in Settings().ollama_base_url

    def test_model_defaults_to_qwen3_8b(self, clean_env: None) -> None:
        settings = Settings()
        assert settings.ollama_model_id == DEFAULT_OLLAMA_MODEL_ID
        assert settings.ollama_model_id == "qwen3:8b"

    def test_request_timeout_default_is_generous(self, clean_env: None) -> None:
        # Local CPU inference is slow, so the default must be well above
        # the 30 s used for the hosted provider.
        settings = Settings()
        assert settings.ollama_request_timeout_seconds == (
            DEFAULT_OLLAMA_REQUEST_TIMEOUT_SECONDS
        )
        assert settings.ollama_request_timeout_seconds >= 60

    def test_connect_timeout_is_short(self, clean_env: None) -> None:
        # A stopped Ollama server must fail fast, not hang.
        assert Settings().ollama_connect_timeout_seconds <= 10

    def test_generation_still_disabled_by_default(self, clean_env: None) -> None:
        assert Settings().ai_generation_enabled is False


class TestProviderSelection:
    """The provider selector is explicit and validated."""

    def test_the_removed_gemini_value_is_still_accepted(
        self, clean_env: None
    ) -> None:
        # The value is still accepted so an existing deployment's
        # ``LLM_PROVIDER`` setting does not become an unparseable value.
        # Selecting it fails closed at provider-construction time,
        # because that backend no longer exists.
        settings = Settings(llm_provider="gemini")
        assert settings.llm_provider == "gemini"

    def test_selecting_the_removed_gemini_backend_fails_closed(
        self, clean_env: None
    ) -> None:
        from app.errors import MissingOllamaConfigError
        from app.providers import build_structured_provider

        with pytest.raises(MissingOllamaConfigError):
            build_structured_provider(Settings(llm_provider="gemini"))

    def test_invalid_provider_is_rejected(self, clean_env: None) -> None:
        with pytest.raises(ValidationError):
            Settings(llm_provider="openai")

    def test_provider_reads_from_env(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        assert Settings().llm_provider == "gemini"

    def test_model_id_reads_from_env(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OLLAMA_MODEL_ID", "qwen3:4b")
        assert Settings().ollama_model_id == "qwen3:4b"

    def test_base_url_reads_from_env(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11435")
        assert Settings().ollama_base_url == "http://127.0.0.1:11435"

    def test_zero_timeout_is_rejected(self, clean_env: None) -> None:
        with pytest.raises(ValidationError):
            Settings(ollama_request_timeout_seconds=0)

    def test_out_of_range_timeout_is_rejected(self, clean_env: None) -> None:
        with pytest.raises(ValidationError):
            Settings(ollama_request_timeout_seconds=10_000)

    def test_temperature_is_bounded(self, clean_env: None) -> None:
        with pytest.raises(ValidationError):
            Settings(ollama_temperature=5.0)


class TestReprSafety:
    """``__repr__`` must not echo secret material."""

    def test_repr_masks_secrets(self, clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "another-secret-67890")
        text = repr(Settings())
        assert "another-secret-67890" not in text
        assert "***" in text

    def test_repr_includes_the_provider(self, clean_env: None) -> None:
        assert "llm_provider='ollama'" in repr(Settings())
