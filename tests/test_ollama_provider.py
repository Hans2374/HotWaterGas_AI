"""Tests for the local Ollama provider.

Every HTTP call is mocked with ``httpx.MockTransport`` — no socket is
ever opened, no model is ever downloaded, and no paid API is ever
called. The tests cover:

* Request construction: model id, ``stream: false``, JSON-Schema
  ``format``, message roles, and the option block.
* Valid structured response parsing (including a fenced code block).
* Malformed JSON, a non-object payload, a missing message, and empty
  content.
* Server unavailable, timeout, missing model, and upstream HTTP errors.
* Sanitised errors: no upstream body, no secret, no stack trace.
* The ``/api/tags`` model check, including its caching behaviour.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import pytest

from app.errors import (
    InvalidProviderOutputError,
    ProviderError,
    ProviderModelNotFoundError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.providers.ollama import CHAT_PATH, TAGS_PATH, OllamaProvider, build_ollama_provider


# ── Helpers ───────────────────────────────────────────────────────────────


def _chat_envelope(content: str) -> Dict[str, Any]:
    """Build an Ollama chat envelope with the given assistant content."""

    return {
        "model": "qwen3:8b",
        "created_at": "2026-10-26T00:00:00Z",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
        "eval_count": 42,
    }


def _provider(
    handler: Any,
    **kwargs: Any,
) -> OllamaProvider:
    """Build a provider wired to a mock transport."""

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    options: Dict[str, Any] = {
        "base_url": "http://127.0.0.1:11434",
        "model_id": "qwen3:8b",
        "request_timeout_seconds": 30,
        "connect_timeout_seconds": 5,
        "tags_timeout_seconds": 10,
    }
    options.update(kwargs)
    return OllamaProvider(client=client, **options)


class _Recorder:
    """Records every request the mock transport receives."""

    def __init__(self) -> None:
        self.requests: List[httpx.Request] = []

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]

    def last_json(self) -> Dict[str, Any]:
        return json.loads(self.last.content.decode("utf-8"))


# ── Constructor ───────────────────────────────────────────────────────────


class TestConstructor:
    """Provider construction rejects invalid configuration."""

    def test_missing_base_url_raises(self) -> None:
        from app.errors import MissingOllamaConfigError

        with pytest.raises(MissingOllamaConfigError):
            OllamaProvider(
                base_url="",
                model_id="qwen3:8b",
                request_timeout_seconds=30,
            )

    def test_missing_model_id_raises(self) -> None:
        from app.errors import MissingOllamaConfigError

        with pytest.raises(MissingOllamaConfigError):
            OllamaProvider(
                base_url="http://127.0.0.1:11434",
                model_id="",
                request_timeout_seconds=30,
            )

    def test_invalid_timeout_raises(self) -> None:
        with pytest.raises(ValueError):
            OllamaProvider(
                base_url="http://127.0.0.1:11434",
                model_id="qwen3:8b",
                request_timeout_seconds=0,
            )

    def test_invalid_connect_timeout_raises(self) -> None:
        with pytest.raises(ValueError):
            OllamaProvider(
                base_url="http://127.0.0.1:11434",
                model_id="qwen3:8b",
                request_timeout_seconds=30,
                connect_timeout_seconds=0,
            )

    def test_base_url_trailing_slash_is_stripped(self) -> None:
        provider = _provider(
            lambda request: httpx.Response(200, json=_chat_envelope("{}")),
            base_url="http://127.0.0.1:11434/",
        )
        assert provider.base_url == "http://127.0.0.1:11434"

    def test_model_id_exposed(self) -> None:
        provider = _provider(
            lambda request: httpx.Response(200, json=_chat_envelope("{}")),
            model_id="qwen3:8b",
        )
        assert provider.model_id == "qwen3:8b"


# ── Request construction ──────────────────────────────────────────────────


class TestRequestConstruction:
    """The outgoing request must match the Ollama HTTP API contract."""

    def test_posts_to_chat_endpoint(self) -> None:
        recorder = _Recorder()

        def handler(request: httpx.Request) -> httpx.Response:
            recorder.requests.append(request)
            return httpx.Response(200, json=_chat_envelope('{"ok": true}'))

        provider = _provider(handler)
        provider.generate_structured(
            system_prompt="SYS",
            user_prompt="USR",
            json_schema={"type": "object"},
        )
        assert str(recorder.last.url) == f"http://127.0.0.1:11434{CHAT_PATH}"
        assert recorder.last.method == "POST"

    def test_body_contains_model_id(self) -> None:
        recorder = _Recorder()

        def handler(request: httpx.Request) -> httpx.Response:
            recorder.requests.append(request)
            return httpx.Response(200, json=_chat_envelope('{"ok": true}'))

        provider = _provider(handler, model_id="qwen3:8b")
        provider.generate_structured(
            system_prompt="SYS",
            user_prompt="USR",
            json_schema={"type": "object"},
        )
        assert recorder.last_json()["model"] == "qwen3:8b"

    def test_stream_is_false(self) -> None:
        recorder = _Recorder()

        def handler(request: httpx.Request) -> httpx.Response:
            recorder.requests.append(request)
            return httpx.Response(200, json=_chat_envelope('{"ok": true}'))

        provider = _provider(handler)
        provider.generate_structured(
            system_prompt="SYS",
            user_prompt="USR",
            json_schema={"type": "object"},
        )
        assert recorder.last_json()["stream"] is False

    def test_format_carries_the_json_schema(self) -> None:
        recorder = _Recorder()
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }

        def handler(request: httpx.Request) -> httpx.Response:
            recorder.requests.append(request)
            return httpx.Response(200, json=_chat_envelope('{"ok": true}'))

        provider = _provider(handler)
        provider.generate_structured(
            system_prompt="SYS",
            user_prompt="USR",
            json_schema=schema,
        )
        assert recorder.last_json()["format"] == schema

    def test_messages_use_system_and_user_roles(self) -> None:
        recorder = _Recorder()

        def handler(request: httpx.Request) -> httpx.Response:
            recorder.requests.append(request)
            return httpx.Response(200, json=_chat_envelope('{"ok": true}'))

        provider = _provider(handler)
        provider.generate_structured(
            system_prompt="SYSTEM PROMPT",
            user_prompt="USER PROMPT",
            json_schema={"type": "object"},
        )
        messages = recorder.last_json()["messages"]
        assert messages == [
            {"role": "system", "content": "SYSTEM PROMPT"},
            {"role": "user", "content": "USER PROMPT"},
        ]

    def test_options_carry_temperature_and_num_predict(self) -> None:
        recorder = _Recorder()

        def handler(request: httpx.Request) -> httpx.Response:
            recorder.requests.append(request)
            return httpx.Response(200, json=_chat_envelope('{"ok": true}'))

        provider = _provider(handler, num_predict=1234, temperature=0.35)
        provider.generate_structured(
            system_prompt="SYS",
            user_prompt="USR",
            json_schema={"type": "object"},
        )
        options = recorder.last_json()["options"]
        assert options["temperature"] == 0.35
        assert options["num_predict"] == 1234

    def test_no_credentials_are_sent(self) -> None:
        recorder = _Recorder()

        def handler(request: httpx.Request) -> httpx.Response:
            recorder.requests.append(request)
            return httpx.Response(200, json=_chat_envelope('{"ok": true}'))

        provider = _provider(handler)
        provider.generate_structured(
            system_prompt="SYS",
            user_prompt="USR",
            json_schema={"type": "object"},
        )
        headers = {k.lower() for k in recorder.last.headers.keys()}
        assert "authorization" not in headers
        assert "x-api-key" not in headers


# ── Response parsing ──────────────────────────────────────────────────────


class TestResponseParsing:
    """Valid structured responses are parsed into a mapping."""

    def test_parses_json_content(self) -> None:
        payload = {"description": {"text": "Xin chào", "word_count": 2}}
        provider = _provider(
            lambda request: httpx.Response(
                200, json=_chat_envelope(json.dumps(payload))
            )
        )
        result = provider.generate_structured(
            system_prompt="SYS",
            user_prompt="USR",
            json_schema={"type": "object"},
        )
        assert result == payload

    def test_parses_fenced_code_block(self) -> None:
        content = '```json\n{"ok": true}\n```'
        provider = _provider(
            lambda request: httpx.Response(200, json=_chat_envelope(content))
        )
        result = provider.generate_structured(
            system_prompt="SYS",
            user_prompt="USR",
            json_schema={"type": "object"},
        )
        assert result == {"ok": True}

    def test_parses_fenced_code_block_without_language(self) -> None:
        content = '```\n{"ok": true}\n```'
        provider = _provider(
            lambda request: httpx.Response(200, json=_chat_envelope(content))
        )
        result = provider.generate_structured(
            system_prompt="SYS",
            user_prompt="USR",
            json_schema={"type": "object"},
        )
        assert result == {"ok": True}


# ── Malformed output ──────────────────────────────────────────────────────


class TestMalformedOutput:
    """Unusable upstream output is an invalid-output error."""

    def test_malformed_json_raises(self) -> None:
        provider = _provider(
            lambda request: httpx.Response(
                200, json=_chat_envelope("{not valid json")
            )
        )
        with pytest.raises(InvalidProviderOutputError):
            provider.generate_structured(
                system_prompt="SYS",
                user_prompt="USR",
                json_schema={"type": "object"},
            )

    def test_non_object_json_raises(self) -> None:
        provider = _provider(
            lambda request: httpx.Response(200, json=_chat_envelope("[1, 2, 3]"))
        )
        with pytest.raises(InvalidProviderOutputError):
            provider.generate_structured(
                system_prompt="SYS",
                user_prompt="USR",
                json_schema={"type": "object"},
            )

    def test_missing_message_raises(self) -> None:
        provider = _provider(
            lambda request: httpx.Response(200, json={"model": "qwen3:8b", "done": True})
        )
        with pytest.raises(InvalidProviderOutputError):
            provider.generate_structured(
                system_prompt="SYS",
                user_prompt="USR",
                json_schema={"type": "object"},
            )

    def test_empty_content_raises(self) -> None:
        provider = _provider(
            lambda request: httpx.Response(200, json=_chat_envelope("   "))
        )
        with pytest.raises(InvalidProviderOutputError):
            provider.generate_structured(
                system_prompt="SYS",
                user_prompt="USR",
                json_schema={"type": "object"},
            )

    def test_non_object_envelope_raises(self) -> None:
        provider = _provider(
            lambda request: httpx.Response(200, json=["not", "an", "object"])
        )
        with pytest.raises(InvalidProviderOutputError):
            provider.generate_structured(
                system_prompt="SYS",
                user_prompt="USR",
                json_schema={"type": "object"},
            )

    def test_non_json_envelope_raises(self) -> None:
        provider = _provider(
            lambda request: httpx.Response(200, text="<html>not json</html>")
        )
        with pytest.raises(InvalidProviderOutputError):
            provider.generate_structured(
                system_prompt="SYS",
                user_prompt="USR",
                json_schema={"type": "object"},
            )


# ── Upstream failures ─────────────────────────────────────────────────────


class TestUpstreamFailures:
    """Transport and HTTP failures map to the domain hierarchy."""

    def test_connection_error_raises_unavailable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        provider = _provider(handler)
        with pytest.raises(ProviderUnavailableError):
            provider.generate_structured(
                system_prompt="SYS",
                user_prompt="USR",
                json_schema={"type": "object"},
            )

    def test_read_timeout_raises_timeout(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow")

        provider = _provider(handler)
        with pytest.raises(ProviderTimeoutError):
            provider.generate_structured(
                system_prompt="SYS",
                user_prompt="USR",
                json_schema={"type": "object"},
            )

    def test_connect_timeout_raises_timeout(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("too slow")

        provider = _provider(handler)
        with pytest.raises(ProviderTimeoutError):
            provider.generate_structured(
                system_prompt="SYS",
                user_prompt="USR",
                json_schema={"type": "object"},
            )

    def test_404_raises_model_not_found(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404, json={"error": "model 'missing:model' not found"}
            )

        provider = _provider(handler)
        with pytest.raises(ProviderModelNotFoundError):
            provider.generate_structured(
                system_prompt="SYS",
                user_prompt="USR",
                json_schema={"type": "object"},
            )

    def test_500_raises_unavailable(self) -> None:
        provider = _provider(
            lambda request: httpx.Response(500, text="internal server error")
        )
        with pytest.raises(ProviderUnavailableError):
            provider.generate_structured(
                system_prompt="SYS",
                user_prompt="USR",
                json_schema={"type": "object"},
            )

    def test_400_raises_unavailable(self) -> None:
        provider = _provider(
            lambda request: httpx.Response(400, text="bad request")
        )
        with pytest.raises(ProviderUnavailableError):
            provider.generate_structured(
                system_prompt="SYS",
                user_prompt="USR",
                json_schema={"type": "object"},
            )

    def test_429_is_mapped_to_unavailable_not_a_rate_limit_error(self) -> None:
        # Documents the *current* behaviour rather than asserting an
        # ideal. The provider maps every status >= 400 to
        # ``ProviderUnavailableError``, so a 429 from a local Ollama
        # server surfaces as 502 and **not** as the 429-mapped
        # ``ProviderRateLimitError``. That class exists for the
        # Gemini path; it is unreachable from this provider.
        #
        # A known open question tracks whether this should change.
        # This test pins the current behaviour so a future change is a
        # deliberate, visible one.
        provider = _provider(
            lambda request: httpx.Response(429, text="too many requests")
        )
        with pytest.raises(ProviderUnavailableError):
            provider.generate_structured(
                system_prompt="SYS",
                user_prompt="USR",
                json_schema={"type": "object"},
            )

    def test_a_429_does_not_raise_the_rate_limit_error(self) -> None:
        # The explicit negative: the rate-limit class is not used here.
        from app.errors import ProviderRateLimitError

        provider = _provider(
            lambda request: httpx.Response(429, text="too many requests")
        )
        with pytest.raises(ProviderError) as excinfo:
            provider.generate_structured(
                system_prompt="SYS",
                user_prompt="USR",
                json_schema={"type": "object"},
            )
        assert not isinstance(excinfo.value, ProviderRateLimitError)


# ── Sanitised errors ──────────────────────────────────────────────────────


class TestSanitisedErrors:
    """Upstream bodies and secrets must never reach the caller."""

    def test_upstream_body_is_not_in_the_error_message(self) -> None:
        secret = "SUPER-SECRET-UPSTREAM-BODY"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text=secret)

        provider = _provider(handler)
        with pytest.raises(ProviderUnavailableError) as excinfo:
            provider.generate_structured(
                system_prompt="SYS",
                user_prompt="USR",
                json_schema={"type": "object"},
            )
        assert secret not in excinfo.value.message
        assert secret not in str(excinfo.value)

    def test_model_id_from_upstream_body_is_not_leaked(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404, json={"error": "model 'secret-model-name' not found"}
            )

        provider = _provider(handler)
        with pytest.raises(ProviderModelNotFoundError) as excinfo:
            provider.generate_structured(
                system_prompt="SYS",
                user_prompt="USR",
                json_schema={"type": "object"},
            )
        assert "secret-model-name" not in excinfo.value.message

    def test_malformed_content_is_not_leaked(self) -> None:
        provider = _provider(
            lambda request: httpx.Response(
                200, json=_chat_envelope("LEAKED-MODEL-OUTPUT {broken")
            )
        )
        with pytest.raises(InvalidProviderOutputError) as excinfo:
            provider.generate_structured(
                system_prompt="SYS",
                user_prompt="USR",
                json_schema={"type": "object"},
            )
        assert "LEAKED-MODEL-OUTPUT" not in excinfo.value.message

    def test_no_stack_trace_in_error_message(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        provider = _provider(handler)
        with pytest.raises(ProviderUnavailableError) as excinfo:
            provider.generate_structured(
                system_prompt="SYS",
                user_prompt="USR",
                json_schema={"type": "object"},
            )
        assert "Traceback" not in excinfo.value.message
        assert "httpx" not in excinfo.value.message


# ── Model availability ────────────────────────────────────────────────────


class TestModelAvailability:
    """The ``/api/tags`` probe detects a missing model."""

    def test_lists_available_models(self) -> None:
        recorder = _Recorder()

        def handler(request: httpx.Request) -> httpx.Response:
            recorder.requests.append(request)
            return httpx.Response(
                200,
                json={"models": [{"name": "qwen3:8b"}, {"name": "llama3:8b"}]},
            )

        provider = _provider(handler)
        models = provider.available_models()
        assert "qwen3:8b" in models
        assert "llama3:8b" in models
        assert str(recorder.last.url) == f"http://127.0.0.1:11434{TAGS_PATH}"

    def test_model_is_available_when_listed(self) -> None:
        provider = _provider(
            lambda request: httpx.Response(
                200, json={"models": [{"name": "qwen3:8b"}]}
            )
        )
        assert provider.is_model_available() is True
        provider.ensure_model_available()

    def test_missing_model_raises_model_not_found(self) -> None:
        provider = _provider(
            lambda request: httpx.Response(
                200, json={"models": [{"name": "llama3:8b"}]}
            )
        )
        assert provider.is_model_available() is False
        with pytest.raises(ProviderModelNotFoundError):
            provider.ensure_model_available()

    def test_tags_are_cached_between_calls(self) -> None:
        calls = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            return httpx.Response(
                200, json={"models": [{"name": "qwen3:8b"}]}
            )

        provider = _provider(handler)
        provider.available_models()
        provider.available_models()
        provider.available_models()
        assert calls["count"] == 1

    def test_tags_unavailable_raises_unavailable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        provider = _provider(handler)
        with pytest.raises(ProviderUnavailableError):
            provider.available_models()

    def test_tags_timeout_raises_timeout(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow")

        provider = _provider(handler)
        with pytest.raises(ProviderTimeoutError):
            provider.available_models()

    def test_tags_http_error_raises_unavailable(self) -> None:
        provider = _provider(lambda request: httpx.Response(503, text="nope"))
        with pytest.raises(ProviderUnavailableError):
            provider.available_models()

    def test_tags_invalid_json_raises_unavailable(self) -> None:
        provider = _provider(lambda request: httpx.Response(200, text="not json"))
        with pytest.raises(ProviderUnavailableError):
            provider.available_models()

    def test_untagged_model_name_is_also_accepted(self) -> None:
        provider = _provider(
            lambda request: httpx.Response(
                200, json={"models": [{"name": "qwen3:8b", "model": "qwen3:8b"}]}
            )
        )
        assert provider.is_model_available() is True


# ── Factory ───────────────────────────────────────────────────────────────


class TestFactory:
    """The factory builds a provider from resolved settings."""

    def test_factory_uses_settings(self) -> None:
        from app.config import Settings

        settings = Settings(
            llm_provider="ollama",
            ollama_base_url="http://127.0.0.1:11434",
            ollama_model_id="qwen3:8b",
            ollama_request_timeout_seconds=45,
        )
        provider = build_ollama_provider(settings)
        assert isinstance(provider, OllamaProvider)
        assert provider.model_id == "qwen3:8b"
        assert provider.base_url == "http://127.0.0.1:11434"
        provider.close()

    def test_factory_rejects_empty_model_id(self) -> None:
        from app.config import Settings
        from app.errors import MissingOllamaConfigError

        settings = Settings(llm_provider="ollama", ollama_model_id="")
        with pytest.raises(MissingOllamaConfigError):
            build_ollama_provider(settings)
