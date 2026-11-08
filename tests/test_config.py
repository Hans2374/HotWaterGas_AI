"""Configuration tests.

These tests pin the safety guarantees of the configuration:

* ``ai_generation_enabled`` defaults to ``False`` when the env var is absent.
* The service starts without any provider credentials.
* Booleans accept a few common spellings.
* The ``Settings.__repr__`` never includes secret values.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings


class TestDefaults:
    """Defaults that keep the service safe in a fresh environment."""

    def test_generation_disabled_by_default(self, clean_env: None) -> None:
        settings = Settings()
        assert settings.ai_generation_enabled is False

    def test_app_env_defaults_to_development(self, clean_env: None) -> None:
        settings = Settings()
        assert settings.app_env == "development"

    def test_internal_token_defaults_to_empty_string(self, clean_env: None) -> None:
        settings = Settings()
        assert settings.internal_service_token == ""

    def test_the_removed_gemini_settings_are_gone(self, clean_env: None) -> None:
        # The Gemini-backed pipeline has been removed, so the service
        # must not expose its settings any more: an operator must not be
        # able to configure a provider that no longer exists.
        settings = Settings()
        assert not hasattr(settings, "gemini_api_key")
        assert not hasattr(settings, "gemini_model_id")
        assert not hasattr(settings, "gemini_request_timeout_seconds")


class TestBooleanCoercion:
    """Accept common boolean spellings from environment variables."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("true", True),
            ("TRUE", True),
            ("True", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("FALSE", False),
            ("0", False),
            ("no", False),
            ("off", False),
            ("", False),
        ],
    )
    def test_coerces_common_spellings(
        self,
        clean_env: None,
        monkeypatch: pytest.MonkeyPatch,
        raw: str,
        expected: bool,
    ) -> None:
        monkeypatch.setenv("AI_GENERATION_ENABLED", raw)
        assert Settings().ai_generation_enabled is expected

    def test_invalid_boolean_raises(self, clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_GENERATION_ENABLED", "maybe")
        with pytest.raises(ValidationError):
            Settings()


class TestStartupSafety:
    """The service can build ``Settings`` without any credentials."""

    def test_no_env_vars_means_safe_defaults(self, clean_env: None) -> None:
        settings = Settings()
        assert settings.ai_generation_enabled is False
        assert settings.ollama_model_id == "qwen3:8b"
        assert settings.internal_service_token == ""


class TestReprSafety:
    """``__repr__`` must never echo secret material."""

    def test_repr_masks_the_internal_token(self, clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "another-secret-67890")
        settings = Settings()
        text = repr(settings)
        assert "another-secret-67890" not in text
        assert "***" in text


class TestRemovedStructuredDraftSettings:
    """The settings of the removed structured-draft pipeline are gone."""

    def test_the_removed_accessors_are_gone(self, clean_env: None) -> None:
        # The structured-draft path has been removed, so an operator
        # must not be able to configure bounds that no longer govern
        # anything.
        settings = Settings()
        for name in (
            "v2_min_target_words",
            "v2_max_target_words",
            "v2_max_description_words",
            "v2_max_product_name_length",
            "v2_max_short_field_length",
            "v2_max_facts_length",
            "v2_max_requirement_value_length",
            "v2_max_list_item_length",
            "v2_max_list_items",
            "v2_max_provider_input_bytes",
            "v2_max_request_bytes",
        ):
            assert not hasattr(settings, name), name

    def test_the_historical_gemini_accessors_are_gone(self, clean_env: None) -> None:
        # The ``gemini_``-prefixed accessors existed only for the
        # structured-draft path. With both the provider and the path
        # gone, they are gone too.
        settings = Settings()
        assert not hasattr(settings, "gemini_max_description_words")
        assert not hasattr(settings, "gemini_min_target_words")
        assert not hasattr(settings, "gemini_max_target_words")

    def test_the_translation_bounds_are_still_available(self, clean_env: None) -> None:
        # The translation bounds are the retained generation path's own
        # bounds.
        settings = Settings()
        assert settings.min_target_words == 500
        assert settings.max_target_words == 600
        assert settings.max_description_words == 800
        assert settings.max_source_bytes == 10 * 1024
        assert settings.max_provider_input_bytes == 32768
        assert settings.max_request_bytes == 8192
        assert settings.min_source_tokens == 350


# ── The legacy environment-variable spellings ─────────────────────────────


class TestLegacyEnvAliases:
    """The old ``V3_*`` spellings still work, and the new ones win.

    The translation bounds were originally spelled with a ``V3_``
    prefix. The names are now neutral, but a deployment that still
    exports the old spelling must keep working: silently falling back
    to the default would quietly change the operator's configured
    limits.
    """

    LEGACY_CASES = {
        "V3_MIN_TARGET_WORDS": ("min_target_words", 111),
        "V3_MAX_TARGET_WORDS": ("max_target_words", 222),
        "V3_MAX_DESCRIPTION_WORDS": ("max_description_words", 333),
        "V3_MAX_SOURCE_BYTES": ("max_source_bytes", 4444),
        "V3_MAX_PROVIDER_INPUT_BYTES": ("max_provider_input_bytes", 55555),
        "V3_MIN_SOURCE_TOKENS": ("min_source_tokens", 66),
    }

    @pytest.mark.parametrize("legacy,case", list(LEGACY_CASES.items()))
    def test_the_legacy_spelling_is_honoured(
        self, legacy: str, case: tuple[str, int], clean_env: None, monkeypatch
    ) -> None:
        field, value = case
        monkeypatch.setenv(legacy, str(value))
        settings = Settings()
        assert getattr(settings, field) == value

    @pytest.mark.parametrize("legacy,case", list(LEGACY_CASES.items()))
    def test_the_neutral_spelling_is_honoured(
        self, legacy: str, case: tuple[str, int], clean_env: None, monkeypatch
    ) -> None:
        field, value = case
        neutral = field.upper()
        monkeypatch.setenv(neutral, str(value))
        settings = Settings()
        assert getattr(settings, field) == value

    @pytest.mark.parametrize("legacy,case", list(LEGACY_CASES.items()))
    def test_the_neutral_spelling_wins_when_both_are_present(
        self, legacy: str, case: tuple[str, int], clean_env: None, monkeypatch
    ) -> None:
        # The neutral name is the one the operator has been told to use,
        # so it takes precedence. The legacy value is ignored.
        field, value = case
        monkeypatch.setenv(legacy, "999")
        monkeypatch.setenv(field.upper(), str(value))
        settings = Settings()
        assert getattr(settings, field) == value

    def test_the_alias_table_covers_every_configurable_bound(self) -> None:
        # A new bound must not be added without deciding whether it
        # needs a legacy spelling.
        from app.config import LEGACY_ENV_ALIASES

        assert set(LEGACY_ENV_ALIASES.values()) == {
            "MIN_TARGET_WORDS",
            "MAX_TARGET_WORDS",
            "MAX_DESCRIPTION_WORDS",
            "MAX_SOURCE_BYTES",
            "MAX_PROVIDER_INPUT_BYTES",
            "MIN_SOURCE_TOKENS",
        }
