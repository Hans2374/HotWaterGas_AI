"""Tests for the domain error hierarchy.

These tests pin the HTTP status / public code mapping for every domain
exception so a future refactor cannot silently change the API
contract.
"""

from __future__ import annotations

import pytest

from app.errors import (
    GenerationDisabledError,
    HotWaterGasAIError,
    InvalidInternalTokenError,
    InvalidProviderOutputError,
    MissingGeminiConfigError,
    MissingInternalTokenConfigError,
    ProviderAuthError,
    ProviderError,
    ProviderModelNotFoundError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RequestTooLargeError,
)


class TestHTTPMapping:
    """Every error class has a stable HTTP status and public code."""

    @pytest.mark.parametrize(
        "exc_cls, expected_status, expected_code",
        [
            (GenerationDisabledError, 503, "generation_disabled"),
            (MissingInternalTokenConfigError, 500, "internal_token_not_configured"),
            (InvalidInternalTokenError, 401, "unauthorized"),
            (MissingGeminiConfigError, 503, "provider_not_configured"),
            (ProviderAuthError, 502, "provider_auth_failed"),
            (ProviderModelNotFoundError, 502, "provider_model_not_found"),
            (ProviderRateLimitError, 429, "rate_limited"),
            (ProviderTimeoutError, 504, "provider_timeout"),
            (ProviderUnavailableError, 502, "provider_unavailable"),
            (InvalidProviderOutputError, 502, "invalid_provider_output"),
            (RequestTooLargeError, 413, "request_too_large"),
        ],
    )
    def test_status_and_code(
        self,
        exc_cls: type,
        expected_status: int,
        expected_code: str,
    ) -> None:
        assert exc_cls.http_status == expected_status
        assert exc_cls.public_code == expected_code

    def test_provider_subclasses_inherit_from_provider_error(self) -> None:
        for cls in (
            ProviderAuthError,
            ProviderModelNotFoundError,
            ProviderRateLimitError,
            ProviderTimeoutError,
            ProviderUnavailableError,
            InvalidProviderOutputError,
        ):
            assert issubclass(cls, ProviderError)
            assert issubclass(cls, HotWaterGasAIError)

    def test_default_message_is_class_name(self) -> None:
        # Useful for the route layer; the public message is
        # never None.
        exc = ProviderTimeoutError()
        assert exc.message == "ProviderTimeoutError"

    def test_custom_message_preserved(self) -> None:
        exc = ProviderTimeoutError("upstream is slow")
        assert exc.message == "upstream is slow"
