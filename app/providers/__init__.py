"""Provider abstractions for the HotWaterGas AI service.

Only this sub-package is allowed to import the underlying provider SDK.
The route and service layers depend on the abstract interfaces in
:mod:`app.providers.base` so a backend can be replaced (or mocked)
without rewriting business logic.

Provider selection
------------------
``Settings.llm_provider`` chooses the backend:

``ollama``
    The local, free, offline default. It is the provider used during
    development and by the test suite. No credential is required and
    no paid API is called.

``gemini``
    A legacy value that is still accepted so an existing deployment's
    setting does not become unparseable, but the backend behind it has
    been removed. Selecting it fails closed.

There is **no automatic fallback** between the two. A failing provider
surfaces as a controlled domain error; the service never silently
switches models or backends.
"""

from __future__ import annotations

from app.config import Settings
from app.providers.base import StructuredProvider
from app.providers.ollama import OllamaProvider, build_ollama_provider


def build_structured_provider(settings: Settings) -> StructuredProvider:
    """Construct the structured provider for the selected backend.

    Only the local Ollama backend implements the structured contract.
    Any other selection is explicitly unsupported rather than silently
    downgraded to free-form text: selecting it is an operator error,
    not a runtime fallback.
    """

    if settings.llm_provider == "ollama":
        return build_ollama_provider(settings)

    from app.errors import MissingOllamaConfigError

    raise MissingOllamaConfigError(
        "The selected provider does not support structured generation."
    )


__all__ = [
    "StructuredProvider",
    "OllamaProvider",
    "build_ollama_provider",
    "build_structured_provider",
]
