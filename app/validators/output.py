"""Deterministic validation of a generated description.

These functions operate on the **text** returned by the provider, never on
raw SDK objects. They:

* strip surrounding whitespace,
* reject empty / whitespace-only / obviously unusable output,
* enforce a hard maximum output length (so we never silently truncate),
* report whether the output sits inside the soft 120–200-word target.

There is no automatic regeneration here — when validation fails, the
orchestrator surfaces a controlled error to the client. The service
explicitly forbids automatic regeneration loops.

The module is **shared**. :func:`count_words` is the word-counting rule
the translation pipeline uses; :func:`validate_generated_text` is the
full validation entry point. The legacy Gemini-backed pipeline that
first introduced them has been removed, but the helpers themselves
are version-neutral and are retained.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ValidatedOutput:
    """The result of validating a provider response.

    Attributes
    ----------
    text:
        The cleaned description text. Always a non-empty string.
    word_count:
        Number of whitespace-separated words in the cleaned text.
    within_target_range:
        ``True`` if the word count is inside the soft target range. This
        is purely informational — the validator does not reject output
        that falls outside the range.
    """

    text: str
    word_count: int
    within_target_range: bool


# Whitespace normalisation. Newlines are collapsed to single spaces, then
# leading and trailing whitespace is stripped. We intentionally do **not**
# remove internal newlines paragraph-by-paragraph because the model is
# supposed to write a small block of text and the consumer (the future
# admin UI) is expected to preserve line breaks if any are present.
_MULTI_WS = re.compile(r"[ \t\f\v]+")


def count_words(text: str) -> int:
    """Count whitespace-separated words.

    Uses ``str.split()`` so the result is deterministic and matches the
    intuitive notion of "word count" for plain text. The empty string
    returns 0.
    """

    return len([w for w in text.split() if w])


def normalize_output(text: str) -> str:
    """Strip and collapse internal whitespace runs.

    Empty input is returned as ``""`` so the caller can decide what to do
    with it.
    """

    if not text:
        return ""
    # Normalise line endings first, then collapse runs of horizontal
    # whitespace while keeping newlines.
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _MULTI_WS.sub(" ", cleaned)
    # Trim each line individually so trailing spaces on a line do not
    # accumulate; then strip the whole result.
    lines = [line.strip() for line in cleaned.split("\n")]
    return "\n".join(lines).strip()


def validate_generated_text(
    raw_text: Optional[str],
    *,
    max_words: int,
    min_target_words: int,
    max_target_words: int,
) -> ValidatedOutput:
    """Validate the text returned by the provider.

    Parameters
    ----------
    raw_text:
        The generated text, or ``None`` if the provider produced no text
        at all.
    max_words:
        Hard upper bound on the number of words the validator will accept.
        Exceeding this raises :class:`InvalidProviderOutputError` so the
        output is **never** silently truncated.
    min_target_words, max_target_words:
        Bounds of the soft 120–200-word target range. The output is
        classified against this range but never rejected on the basis of
        word count alone.
    """

    # Lazy import to avoid a circular dependency with ``app.errors`` at
    # import time (errors.py does not depend on validators, but the
    # import order in some test environments loads validators first).
    from app.errors import InvalidProviderOutputError

    if raw_text is None:
        raise InvalidProviderOutputError("provider returned no text")
    if not isinstance(raw_text, str):
        # Defensive: an upstream that returns, say, a list of segments
        # should not propagate as-is.
        raise InvalidProviderOutputError("provider returned non-text content")

    cleaned = normalize_output(raw_text)
    if not cleaned:
        raise InvalidProviderOutputError("provider returned empty text")

    word_count = count_words(cleaned)
    if word_count == 0:
        raise InvalidProviderOutputError("provider returned whitespace-only text")

    if word_count > max_words:
        # The spec requires safe handling: do not silently truncate.
        raise InvalidProviderOutputError(
            f"provider output exceeds the {max_words}-word maximum"
        )

    within_range = min_target_words <= word_count <= max_target_words

    return ValidatedOutput(
        text=cleaned,
        word_count=word_count,
        within_target_range=within_range,
    )
