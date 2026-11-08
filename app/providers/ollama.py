"""Local Ollama provider for structured generation.

Ollama runs on the developer's machine. It is free, offline, and needs
no credential, so it is the default provider for development and for
the test suite. This module is the only place that knows the Ollama
HTTP API exists; the service and route layers depend on the abstract
:class:`StructuredProvider` contract instead.

Design notes
------------
* **No paid fallback.** The provider never falls back to another
  backend. If Ollama is unavailable the call fails with a controlled
  domain error.
* **Structured output.** ``/api/chat`` receives a JSON Schema in its
  ``format`` field, so the model is constrained to the requested
  shape. This is *structural* assistance only: a schema-valid document
  is still an unverified draft, never a verified fact.
* **Deterministic handling.** ``stream`` is always ``false`` so the
  response is a single JSON document.
* **Bounded time.** Separate connect and request timeouts, plus a
  short timeout for the ``/api/tags`` availability probe.
* **No leakage.** Upstream response bodies, model output, prompts and
  stack traces are never returned to the caller. Only the exception
  *type* and the HTTP status code are logged.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from app.config import Settings
from app.errors import (
    InvalidProviderOutputError,
    ProviderModelNotFoundError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.providers.base import StructuredProvider


logger = logging.getLogger(__name__)


#: Ollama's chat endpoint, relative to the configured base URL.
CHAT_PATH = "/api/chat"

#: Ollama's model-listing endpoint, relative to the configured base URL.
TAGS_PATH = "/api/tags"


class OllamaProvider:
    """A thin, deterministic wrapper around the local Ollama HTTP API."""

    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        request_timeout_seconds: int,
        connect_timeout_seconds: int = 5,
        tags_timeout_seconds: int = 10,
        num_predict: int = 2048,
        temperature: float = 0.2,
        client: Optional[httpx.Client] = None,
    ) -> None:
        if not base_url:
            from app.errors import MissingOllamaConfigError

            raise MissingOllamaConfigError("Ollama base URL is not configured.")
        if not model_id:
            from app.errors import MissingOllamaConfigError

            raise MissingOllamaConfigError("Ollama model id is not configured.")
        if request_timeout_seconds < 1:
            raise ValueError("request_timeout_seconds must be at least 1")
        if connect_timeout_seconds < 1:
            raise ValueError("connect_timeout_seconds must be at least 1")
        if tags_timeout_seconds < 1:
            raise ValueError("tags_timeout_seconds must be at least 1")

        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._request_timeout = httpx.Timeout(
            request_timeout_seconds,
            connect=connect_timeout_seconds,
        )
        self._tags_timeout = httpx.Timeout(
            tags_timeout_seconds,
            connect=connect_timeout_seconds,
        )
        self._num_predict = max(1, int(num_predict))
        self._temperature = float(temperature)

        # A reusable client keeps the TCP connection warm across calls,
        # which matters for a local server. Tests inject their own
        # transport instead of patching the socket layer.
        self._client = client if client is not None else httpx.Client(
            timeout=self._request_timeout,
            follow_redirects=False,
        )
        self._owns_client = client is None

        # The availability probe is cached: checking ``/api/tags`` on
        # every request would double the local round-trips for no
        # benefit. The cache is invalidated when a call fails with a
        # "model not found" error so a freshly-pulled model is picked up
        # without a restart.
        self._known_models: Optional[set[str]] = None

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def model_id(self) -> str:
        """The configured model id. Exposed for response metadata only."""
        return self._model_id

    @property
    def base_url(self) -> str:
        """The configured base URL. Never includes credentials."""
        return self._base_url

    def available_models(self) -> List[str]:
        """Return the model ids the local server reports.

        The result is cached after the first successful call. Raises a
        domain error when the server cannot be reached.
        """

        if self._known_models is not None:
            return sorted(self._known_models)

        try:
            response = self._client.get(
                f"{self._base_url}{TAGS_PATH}",
                timeout=self._tags_timeout,
            )
        except httpx.TimeoutException as exc:
            logger.warning("ollama_tags_timeout")
            raise ProviderTimeoutError(
                "The local Ollama server did not respond in time."
            ) from exc
        except httpx.HTTPError as exc:
            logger.warning("ollama_tags_unavailable error_type=%s", type(exc).__name__)
            raise ProviderUnavailableError(
                "The local Ollama server is currently unavailable."
            ) from exc

        if response.status_code != 200:
            # The body is not read: it may contain paths or other
            # host-specific detail that must not reach the caller.
            logger.warning("ollama_tags_http_error status=%s", response.status_code)
            raise ProviderUnavailableError(
                "The local Ollama server is currently unavailable."
            )

        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("ollama_tags_invalid_json")
            raise ProviderUnavailableError(
                "The local Ollama server returned an unreadable response."
            ) from exc

        models = self._parse_tags(payload)
        self._known_models = set(models)
        return sorted(self._known_models)

    def is_model_available(self) -> bool:
        """Return ``True`` when the configured model is present locally."""
        return self._model_id in set(self.available_models())

    def ensure_model_available(self) -> None:
        """Raise when the configured model is not present locally.

        The check is cheap because :meth:`available_models` is cached.
        """

        if self.is_model_available():
            return
        logger.warning("ollama_model_not_found model_id=%s", self._model_id)
        raise ProviderModelNotFoundError(
            "The configured local model is not installed on the Ollama server."
        )

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Call Ollama and return the parsed structured payload.

        Raises a :class:`app.errors.ProviderError` subclass on failure.
        """

        payload = {
            "model": self._model_id,
            "stream": False,
            "format": json_schema,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": self._temperature,
                "num_predict": self._num_predict,
            },
        }

        try:
            response = self._client.post(
                f"{self._base_url}{CHAT_PATH}",
                json=payload,
                timeout=self._request_timeout,
            )
        except httpx.TimeoutException as exc:
            logger.warning("ollama_request_timeout")
            raise ProviderTimeoutError(
                "The local Ollama server did not respond in time."
            ) from exc
        except httpx.HTTPError as exc:
            logger.warning(
                "ollama_request_failed error_type=%s", type(exc).__name__
            )
            raise ProviderUnavailableError(
                "The local Ollama server is currently unavailable."
            ) from exc

        if response.status_code == 404:
            # Ollama answers a missing model with 404 and a body that
            # echoes the model id. The body is not propagated.
            logger.warning("ollama_model_not_found model_id=%s", self._model_id)
            # The model list may be stale; force a refresh next time.
            self._known_models = None
            raise ProviderModelNotFoundError(
                "The configured local model is not installed on the Ollama server."
            )

        if response.status_code >= 400:
            logger.warning("ollama_http_error status=%s", response.status_code)
            raise ProviderUnavailableError(
                "The local Ollama server returned an error."
            )

        try:
            envelope = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("ollama_invalid_json_envelope")
            raise InvalidProviderOutputError(
                "The local Ollama server returned an unreadable response."
            ) from exc

        return self._extract_payload(envelope)

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_tags(payload: Any) -> List[str]:
        """Extract the model ids from an ``/api/tags`` response."""

        if not isinstance(payload, dict):
            return []
        models = payload.get("models")
        if not isinstance(models, list):
            return []
        names: List[str] = []
        for entry in models:
            if isinstance(entry, dict):
                name = entry.get("name")
                if isinstance(name, str) and name:
                    names.append(name)
                    # Ollama also reports the untagged name; accept it so
                    # ``qwen3:8b`` matches a model pulled as ``qwen3``.
                    model = entry.get("model")
                    if isinstance(model, str) and model and model not in names:
                        names.append(model)
            elif isinstance(entry, str) and entry:
                names.append(entry)
        return names

    def _extract_payload(self, envelope: Any) -> Dict[str, Any]:
        """Pull the JSON payload out of the Ollama chat envelope.

        The envelope looks like
        ``{"message": {"role": "assistant", "content": "<json text>"}}``.
        The content is parsed and returned as a mapping. A malformed
        envelope, a missing message, or content that is not valid JSON
        raises :class:`InvalidProviderOutputError`.
        """

        if not isinstance(envelope, dict):
            logger.warning("ollama_envelope_not_object")
            raise InvalidProviderOutputError(
                "The local Ollama server returned an unreadable response."
            )

        message = envelope.get("message")
        if not isinstance(message, dict):
            logger.warning("ollama_envelope_missing_message")
            raise InvalidProviderOutputError(
                "The local Ollama server returned an unreadable response."
            )

        content = message.get("content")
        if not isinstance(content, str):
            logger.warning("ollama_envelope_missing_content")
            raise InvalidProviderOutputError(
                "The local Ollama server returned no content."
            )

        text = content.strip()
        if not text:
            logger.warning("ollama_envelope_empty_content")
            raise InvalidProviderOutputError(
                "The local Ollama server returned no content."
            )

        # Some models wrap the JSON in a markdown code fence even when
        # a schema is supplied. Strip the fence before parsing.
        if text.startswith("```"):
            newline = text.find("\n")
            if newline != -1:
                text = text[newline + 1 :]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
            text = text.strip()

        try:
            parsed = json.loads(text)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("ollama_content_invalid_json")
            raise InvalidProviderOutputError(
                "The local Ollama server returned malformed JSON."
            ) from exc

        if not isinstance(parsed, dict):
            logger.warning("ollama_content_not_object")
            raise InvalidProviderOutputError(
                "The local Ollama server returned a malformed payload."
            )

        return parsed

    def close(self) -> None:
        """Close the underlying HTTP client if this provider owns it."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "OllamaProvider":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


# ── Factory ───────────────────────────────────────────────────────────────


def build_ollama_provider(settings: Settings) -> OllamaProvider:
    """Construct an :class:`OllamaProvider` from resolved settings."""

    return OllamaProvider(
        base_url=settings.ollama_base_url,
        model_id=settings.ollama_model_id,
        request_timeout_seconds=settings.ollama_request_timeout_seconds,
        connect_timeout_seconds=settings.ollama_connect_timeout_seconds,
        tags_timeout_seconds=settings.ollama_tags_timeout_seconds,
        num_predict=settings.ollama_num_predict,
        temperature=settings.ollama_temperature,
    )


__all__ = [
    "CHAT_PATH",
    "TAGS_PATH",
    "StructuredProvider",
    "OllamaProvider",
    "build_ollama_provider",
]
