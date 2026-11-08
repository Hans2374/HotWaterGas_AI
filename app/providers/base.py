"""Provider contracts shared by every backend adapter.

The protocols live in their own module so the concrete adapter
(``ollama.py``) and the factory in ``app/providers/__init__.py`` can
import them without a circular import.

One contract exists:

``StructuredProvider``
    The contract every retained generation route uses: it returns a
    parsed, schema-shaped mapping. Schema-constrained output is
    *structural* assistance only — it guarantees the shape of the
    document, never the truth of its contents.

The legacy free-text ``DescriptionProvider`` contract, which the
removed Gemini-backed pipeline used, is gone: no retained route asks
a provider for an unstructured description string.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class StructuredProvider(Protocol):
    """Minimal contract for a provider that returns a structured draft.

    Implementations MUST raise a :class:`app.errors.ProviderError`
    subclass on failure. They MUST NOT raise raw transport exceptions:
    the route layer only knows about our own hierarchy.
    """

    @property
    def model_id(self) -> str:
        """The model id that produced the result."""
        ...

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return the parsed structured payload, or raise a domain error."""
        ...


__all__ = ["StructuredProvider"]
