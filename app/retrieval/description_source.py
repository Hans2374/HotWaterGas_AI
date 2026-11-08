"""Selection and parsing of the English Steam description source.

This module is the **description-source** boundary. It answers one
question: *which English text should be translated, and in what shape?*

Everything here is a pure function of the Steam ``appdetails`` payload.
There is no network access, no clock and no model, so every rule can be
tested offline against a sanitized fixture.

Why a dedicated parser
----------------------
A plain ``strip_html`` helper is the wrong tool here. ``strip_html``
collapses every tag into whitespace and returns a
single flat string, which is exactly right for a *requirement value*
(a label and a value on one line) and exactly wrong for a *description*
(a document with headings, paragraphs and lists). Translating a flattened
string loses the structure the source had, and the structure is part of
what has to survive the translation.

So this module parses the markup into **blocks**:

* ``heading`` - an ``<h1>``..``<h6>`` element,
* ``paragraph`` - a ``<p>`` (or a bare run of text),
* ``list_item`` - an ``<li>`` element.

A ``<br>`` is a **block boundary**: it ends the current block and starts
a new one. That is how Steam separates its paragraphs, and it keeps a
line break meaningful without inventing a block type for it.

The source policy
-----------------
Steam offers three description fields, and they are not interchangeable:

``detailed_description``
    The **primary** source. It is the long-form store description.
``about_the_game``
    A **supplementary** source. It is used as the primary only when the
    detailed description is absent. When it is present *and* it carries
    content the detailed description does not, that content is appended
    — but only content that is *demonstrably* new.
``short_description``
    A **supporting** source: one short paragraph of context. It is
    never appended blindly.

Deduplication is **conservative by construction**
------------------------------------------------
A block is considered a duplicate only when its text is *identical* to
a block that was already kept, after whitespace is collapsed and case is
folded. There is no similarity threshold, no fuzzy match and no
token-overlap heuristic: an aggressive similarity rule would silently
delete distinct content, and deleted content is content the translation
never had.

The one place a containment check is used is the short description. A
short description that is *contained* in a block that was already kept
carries no new information — it is a summary of text that is already
there — so it is skipped. The check runs in that direction only (the
short text inside a kept block), never the other way round, because a
kept block being contained in a shorter summary would mean the summary
is the more informative text, which is not how these fields relate.

Truncation
----------
The source is bounded by a byte budget so the assembled prompt always
fits inside the provider's input limit. When the budget bites, the
source is cut **at whole block boundaries**, in source order, and the
fact that it was cut is recorded. A sentence is never cut in half, and
the response never claims that the whole description was translated.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.config import DEFAULT_MAX_SOURCE_BYTES, DEFAULT_MIN_SOURCE_TOKENS

#: The kind of a parsed block.
BLOCK_KIND_HEADING = "heading"
BLOCK_KIND_PARAGRAPH = "paragraph"
BLOCK_KIND_LIST_ITEM = "list_item"

#: The role a block plays in the assembled source.
ROLE_PRIMARY = "primary"
ROLE_SUPPLEMENTARY = "supplementary"
ROLE_SUPPORTING = "supporting"

#: The Steam fields the source policy consults, in priority order.
SOURCE_FIELD_DETAILED = "detailed_description"
SOURCE_FIELD_ABOUT = "about_the_game"
SOURCE_FIELD_SHORT = "short_description"

#: A whitespace run.
_WS_RE = re.compile(r"\s+")

#: Trailing punctuation at the end of a block.
_TRAILING_PUNCT_RE = re.compile(r"[\s\.,;:!?\-–—…]+$")

#: HTML elements that open a paragraph-level block.
_PARAGRAPH_TAGS = frozenset({"p", "div", "section", "article"})

#: HTML elements that open a list item.
_LIST_ITEM_TAG = "li"

#: HTML elements that open a list.
_LIST_TAGS = frozenset({"ul", "ol"})

#: The HTML elements that open a heading, and their level.
_HEADING_TAGS: Dict[str, int] = {
    "h1": 1,
    "h2": 2,
    "h3": 3,
    "h4": 4,
    "h5": 5,
    "h6": 6,
}


def normalize_block_text(text: str) -> str:
    """Return the comparison form of a block's text.

    Whitespace is collapsed and case is folded. Nothing else changes:
    digits, units and punctuation survive, so two blocks that differ in
    substance never collapse into one.

    Trailing punctuation is dropped, because Steam is inconsistent
    about it: one field may end a summary with a full stop and another
    may not. Without this, two spellings of the same sentence would
    look like two different blocks, and the deduplication would let a
    duplicate through. Only *trailing* punctuation is dropped - a
    difference in the middle of a sentence is a real difference.
    """

    collapsed = _WS_RE.sub(" ", str(text)).strip().casefold()
    return _TRAILING_PUNCT_RE.sub("", collapsed)


@dataclass(frozen=True)
class DescriptionBlock:
    """One structural unit of a Steam description.

    Attributes
    ----------
    kind:
        ``"heading"``, ``"paragraph"`` or ``"list_item"``.
    text:
        The block's plain text, with its internal whitespace collapsed.
    level:
        The heading level (1-6). ``None`` for a non-heading block.
    ordered:
        ``True`` when the block sits inside an ordered list.
    role:
        Which source the block came from: ``"primary"``,
        ``"supplementary"`` or ``"supporting"``.
    """

    kind: str
    text: str
    level: Optional[int] = None
    ordered: bool = False
    role: str = ROLE_PRIMARY

    def render(self) -> str:
        """Return the block as one line of the prompt's source text.

        The rendering is deliberately simple and stable: a heading keeps
        its level as a run of ``#`` characters, a list item keeps its
        marker, and a paragraph is rendered as its own line. The model
        is told to preserve this structure in the translation.
        """

        if self.kind == BLOCK_KIND_HEADING:
            level = self.level or 1
            return f"{'#' * level} {self.text}"
        if self.kind == BLOCK_KIND_LIST_ITEM:
            return f"1. {self.text}" if self.ordered else f"- {self.text}"
        return self.text

    def byte_size(self) -> int:
        """Return the size of the rendered block, in UTF-8 bytes."""

        return len(self.render().encode("utf-8"))


@dataclass(frozen=True)
class DescriptionSelection:
    """The outcome of the description-source policy.

    Attributes
    ----------
    blocks:
        The blocks that will be translated, in source order.
    primary_field:
        The Steam field the primary blocks came from. ``None`` when no
        source carried any content at all.
    supplementary_fields:
        The Steam fields that contributed additional blocks.
    skipped_fields:
        The Steam fields that were consulted but contributed nothing,
        because their content was already present.
    truncated:
        ``True`` when the source budget forced blocks to be dropped.
    dropped_blocks:
        The number of blocks that were dropped by the budget.
    total_bytes:
        The size of every parsed block, before the budget was applied.
    used_bytes:
        The size of the blocks that were kept.
    """

    blocks: List[DescriptionBlock] = field(default_factory=list)
    primary_field: Optional[str] = None
    supplementary_fields: List[str] = field(default_factory=list)
    skipped_fields: List[str] = field(default_factory=list)
    truncated: bool = False
    dropped_blocks: int = 0
    total_bytes: int = 0
    used_bytes: int = 0

    @property
    def has_source(self) -> bool:
        """``True`` when at least one block was selected."""

        return bool(self.blocks)

    @property
    def source_text(self) -> str:
        """Return the rendered source, one block per line."""

        return "\n".join(block.render() for block in self.blocks)

    @property
    def source_token_count(self) -> int:
        """The number of whitespace-separated tokens in the rendered source.

        This is the **source-side** length measurement. It counts the
        English text the model is shown, so it says nothing about how
        long the translation will be.
        """

        return len([word for word in self.source_text.split() if word])


@dataclass(frozen=True)
class SourceSufficiency:
    """Whether the selected source can plausibly support the target length.

    The assessment is made on the **source**, never on the model's
    output. A translation that came out short does not prove that the
    source was short, and a translation that came out long does not
    prove that the source was long.

    Attributes
    ----------
    sufficient:
        ``True`` when the selected source carries at least
        ``min_source_tokens`` whitespace-separated tokens.
    source_tokens:
        The measured token count of the selected source.
    min_source_tokens:
        The threshold the count was compared against.
    truncated:
        ``True`` when the source budget dropped blocks. A truncated
        source is *not* sufficient for the target: the material that
        was dropped is exactly the material that would have carried the
        translation past the target, so the assessment accounts for it
        rather than ignoring it.
    """

    sufficient: bool
    source_tokens: int
    min_source_tokens: int
    truncated: bool

    @property
    def reason(self) -> str:
        """A short, non-sensitive explanation of the assessment."""

        if self.truncated:
            return (
                "the source was cut by the source budget before the "
                "threshold was reached"
            )
        if self.sufficient:
            return "the source carries enough material for the target range"
        return "the source carries less material than the target range needs"


# ── The HTML parser ───────────────────────────────────────────────────────


class _DescriptionHTMLParser(HTMLParser):
    """Turn a Steam description fragment into a list of blocks.

    The parser is intentionally small. It knows about headings,
    paragraphs, list items and line breaks, and it ignores everything
    else: inline markup such as ``<strong>`` contributes its text but
    opens no block of its own.

    Text is accumulated per block, so a heading that contains inline
    markup yields one heading with the markup's text inside it - which
    is what a translator needs.

    Standalone bold text
    --------------------
    Steam does not always use a real heading element. A standalone
    section title is frequently written as
    ``<strong>Product Features</strong><br>`` - bold text that is the
    whole of its own block. Treating it as an ordinary paragraph would
    lose the section structure the source clearly has, so a block whose
    entire text is bold is recognised as a **heading**.

    The rule is deliberately narrow: the bold markup must cover the
    *whole* block. A paragraph that merely contains a bold phrase in
    the middle stays a paragraph.
    """

    #: The inline elements Steam uses to mark a standalone section title.
    _BOLD_TAGS = frozenset({"strong", "b"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: List[DescriptionBlock] = []
        # Stack of open block frames: ``{"tag", "kind", "level"}``.
        self._frames: List[Dict[str, Any]] = []
        # Stack of open list frames: ``True`` for an ordered list.
        self._lists: List[bool] = []
        self._buffer: List[str] = []
        # The text collected inside bold markup for the current block.
        self._bold_text: List[str] = []
        self._bold_depth = 0
        self._bold_start = 0

    # ── Helpers ──────────────────────────────────────────────────────

    def _current_kind(self) -> str:
        """Return the kind of the innermost open block."""

        for frame in reversed(self._frames):
            if frame["kind"]:
                return frame["kind"]
        return BLOCK_KIND_PARAGRAPH

    def _current_level(self) -> Optional[int]:
        """Return the heading level of the innermost open heading."""

        for frame in reversed(self._frames):
            if frame["kind"] == BLOCK_KIND_HEADING:
                return frame["level"]
        return None

    def _current_ordered(self) -> bool:
        """Return ``True`` when the innermost open list is ordered."""

        return bool(self._lists and self._lists[-1])

    def _flush(self) -> None:
        """Close the current block, if it carries any text."""

        text = _WS_RE.sub(" ", "".join(self._buffer)).strip()
        bold = _WS_RE.sub(" ", "".join(self._bold_text)).strip()
        self._buffer = []
        self._bold_text = []
        if not text:
            return
        kind = self._current_kind()
        level = self._current_level()
        if kind == BLOCK_KIND_PARAGRAPH and bold and bold == text:
            # The whole block was bold markup: Steam wrote it as a
            # section title, so it is a heading.
            kind = BLOCK_KIND_HEADING
            level = 2
        self.blocks.append(
            DescriptionBlock(
                kind=kind,
                text=text,
                level=level if kind == BLOCK_KIND_HEADING else None,
                ordered=self._current_ordered() if kind == BLOCK_KIND_LIST_ITEM else False,
            )
        )

    def _close_top(self, tag: str) -> None:
        """Close the innermost open frame when it matches ``tag``."""

        if not self._frames:
            return
        if self._frames[-1]["tag"] != tag:
            # A stray end tag: ignore it rather than unwinding a
            # structure that Steam did not write.
            return
        self._flush()
        self._frames.pop()

    # ── HTMLParser hooks ─────────────────────────────────────────────

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        tag = tag.lower()

        if tag == "br":
            # A line break is a block boundary: the current block ends
            # and a new one of the same kind begins.
            self._flush()
            return

        if tag in _LIST_TAGS:
            self._lists.append(tag == "ol")
            self._frames.append({"tag": tag, "kind": None, "level": None})
            return

        if tag == _LIST_ITEM_TAG:
            self._flush()
            self._frames.append(
                {"tag": tag, "kind": BLOCK_KIND_LIST_ITEM, "level": None}
            )
            return

        if tag in _HEADING_TAGS:
            self._flush()
            self._frames.append(
                {
                    "tag": tag,
                    "kind": BLOCK_KIND_HEADING,
                    "level": _HEADING_TAGS[tag],
                }
            )
            return

        if tag in _PARAGRAPH_TAGS:
            self._flush()
            self._frames.append(
                {"tag": tag, "kind": BLOCK_KIND_PARAGRAPH, "level": None}
            )
            return

        if tag in self._BOLD_TAGS:
            # Bold markup is inline, but its text is recorded separately
            # so a block that is *entirely* bold can be recognised as a
            # standalone section title.
            if self._bold_depth == 0:
                self._bold_start = len(self._buffer)
            self._bold_depth += 1
            return

        # Everything else (``em``, ``span``, ``img``, ...) is inline: its
        # text joins the current block.

    def handle_startendtag(self, tag: str, attrs: Any) -> None:
        # A self-closing tag carries no text, so it changes nothing.
        return

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag in self._BOLD_TAGS:
            if self._bold_depth > 0:
                self._bold_depth -= 1
                if self._bold_depth == 0:
                    self._bold_text.append("".join(self._buffer[self._bold_start:]))
            return

        if tag in _LIST_TAGS:
            self._flush()
            if self._lists:
                self._lists.pop()
            for index in range(len(self._frames) - 1, -1, -1):
                if self._frames[index]["tag"] == tag:
                    del self._frames[index]
                    break
            return

        self._close_top(tag)

    def handle_data(self, data: str) -> None:
        self._buffer.append(data)

    def close(self) -> None:  # type: ignore[override]
        super().close()
        self._flush()


def parse_description_html(raw: Any) -> List[DescriptionBlock]:
    """Parse a Steam description fragment into structural blocks.

    Returns an empty list when the input is absent, is not a string, or
    carries no text at all. A missing description is therefore a
    missing description - never an empty block that would later be
    mistaken for real content.
    """

    if not isinstance(raw, str):
        return []
    if not raw.strip():
        return []

    parser = _DescriptionHTMLParser()
    try:
        parser.feed(raw)
        parser.close()
    except Exception:  # noqa: BLE001 - malformed markup must not crash
        return []
    return parser.blocks


def parse_description_text(raw: Any) -> List[DescriptionBlock]:
    """Parse a *plain-text* Steam description field into one block.

    ``short_description`` is served as plain text, but Steam still
    escapes the HTML entities in it (``&quot;``, ``&amp;``). The entities
    are decoded here, exactly as the HTML parser decodes them, so both
    description paths deliver the same readable text.

    The text is wrapped in a single paragraph block so it can travel
    through the same pipeline as the HTML fields, without pretending it
    had a structure it does not have.
    """

    if not isinstance(raw, str):
        return []
    text = _WS_RE.sub(" ", html.unescape(raw)).strip()
    if not text:
        return []
    return [DescriptionBlock(kind=BLOCK_KIND_PARAGRAPH, text=text)]


def _as_steam_data(payload: Any) -> Dict[str, Any]:
    """Return the Steam ``data`` object of a payload.

    Three shapes are accepted:

    * the raw ``data`` object (``{"name": ..., "detailed_description": ...}``),
    * the whole ``appdetails`` envelope (``{"620": {"success": true, "data": {...}}}``),
    * an ``{"data": {...}}`` wrapper.

    Anything else yields an empty mapping, so a caller that passes the
    wrong thing gets "no source" rather than a crash.
    """

    if not isinstance(payload, dict):
        return {}

    if "data" in payload and isinstance(payload.get("data"), dict):
        return payload["data"]

    # The envelope shape: every value is an entry with a "data" object.
    values = list(payload.values())
    if values and all(isinstance(value, dict) for value in values):
        extracted = _extract_data(payload)
        if extracted:
            return extracted

    # Otherwise it is assumed to be the raw data object itself.
    return payload


def _contains_whole_words(needle: str, haystack: str) -> bool:
    """Return ``True`` when ``needle`` appears in ``haystack`` as a phrase.

    The check runs on **whole words**. A naive substring test would
    report that "the perpetual testing" is contained in "...testing
    initiative..." merely because the letters happen to sit next to
    each other, and a summary that is *not* actually present in the
    source would then be dropped as a duplicate.

    Both arguments are already normalised (whitespace collapsed, case
    folded, trailing punctuation removed), so the comparison is a plain
    phrase match on word boundaries.
    """

    if not needle:
        return False
    return f" {needle} " in f" {haystack} "


# ── The source policy ─────────────────────────────────────────────────────


def _extract_data(payload: Any, app_id: Optional[int] = None) -> Dict[str, Any]:
    """Return the ``data`` object of a Steam ``appdetails`` payload.

    The App ID key is looked up first. A payload that holds exactly one
    entry with a ``data`` object is accepted as a fallback, but only when
    it is unambiguous.
    """

    if not isinstance(payload, dict):
        return {}

    if app_id is not None:
        entry = payload.get(str(app_id))
        if isinstance(entry, dict) and isinstance(entry.get("data"), dict):
            return entry["data"]

    candidates = [
        value
        for value in payload.values()
        if isinstance(value, dict) and isinstance(value.get("data"), dict)
    ]
    if len(candidates) == 1:
        return candidates[0]["data"]
    return {}


def select_description_source(
    data: Any,
    *,
    max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
) -> DescriptionSelection:
    """Apply the description-source policy to a Steam payload.

    ``data`` may be the raw Steam ``data`` object or a whole
    ``appdetails`` envelope; both are accepted.

    The policy is documented in the module docstring. In short:
    ``detailed_description`` is the primary source, ``about_the_game``
    contributes only genuinely new content and is the fallback when the
    primary is absent, and ``short_description`` contributes only a
    summary that is not already present.

    The result never contains a partially-served block: the budget is
    applied at block boundaries, and the fact that it bit is recorded in
    ``truncated`` / ``dropped_blocks``.
    """

    if max_source_bytes < 1:
        raise ValueError("max_source_bytes must be positive")

    steam_data = _as_steam_data(data)
    detailed = parse_description_html(steam_data.get(SOURCE_FIELD_DETAILED))
    about = parse_description_html(steam_data.get(SOURCE_FIELD_ABOUT))
    short = parse_description_text(steam_data.get(SOURCE_FIELD_SHORT))

    primary_field: Optional[str] = None
    primary_blocks: List[DescriptionBlock] = []
    supplementary_fields: List[str] = []
    skipped_fields: List[str] = []

    if detailed:
        primary_field = SOURCE_FIELD_DETAILED
        primary_blocks = detailed
    elif about:
        # Fallback: the detailed description is absent, so the about
        # text is the primary source rather than a supplement.
        primary_field = SOURCE_FIELD_ABOUT
        primary_blocks = about

    kept: List[DescriptionBlock] = list(primary_blocks)
    kept_keys = {normalize_block_text(block.text) for block in kept}

    # ── about_the_game as a supplement ──────────────────────────────
    if about and primary_field == SOURCE_FIELD_DETAILED:
        distinct = [
            block
            for block in about
            if normalize_block_text(block.text) not in kept_keys
        ]
        if distinct:
            for block in distinct:
                kept.append(
                    DescriptionBlock(
                        kind=block.kind,
                        text=block.text,
                        level=block.level,
                        ordered=block.ordered,
                        role=ROLE_SUPPLEMENTARY,
                    )
                )
                kept_keys.add(normalize_block_text(block.text))
            supplementary_fields.append(SOURCE_FIELD_ABOUT)
        else:
            skipped_fields.append(SOURCE_FIELD_ABOUT)
    elif about and primary_field == SOURCE_FIELD_ABOUT:
        supplementary_fields.append(SOURCE_FIELD_ABOUT)

    # ── short_description as supporting context ─────────────────────
    if short:
        short_key = normalize_block_text(short[0].text)
        contained = any(
            _contains_whole_words(short_key, normalize_block_text(block.text))
            for block in kept
        )
        if short_key in kept_keys or contained:
            skipped_fields.append(SOURCE_FIELD_SHORT)
        else:
            kept.append(
                DescriptionBlock(
                    kind=BLOCK_KIND_PARAGRAPH,
                    text=short[0].text,
                    role=ROLE_SUPPORTING,
                )
            )
            kept_keys.add(short_key)
            supplementary_fields.append(SOURCE_FIELD_SHORT)

    # ── the source budget ───────────────────────────────────────────
    total_bytes = sum(block.byte_size() for block in kept)
    selected: List[DescriptionBlock] = []
    used = 0
    dropped = 0
    truncated = False

    for block in kept:
        size = block.byte_size()
        if selected and used + size > max_source_bytes:
            # The budget is exhausted. The block is dropped whole and
            # the source is marked as truncated.
            truncated = True
            dropped += 1
            continue
        selected.append(block)
        used += size

    if dropped == 0 and used > max_source_bytes:
        # A single block is larger than the whole budget. It is kept
        # anyway - dropping it would leave nothing to translate - and
        # the source is marked as truncated so the response never claims
        # the whole description was translated.
        truncated = True

    return DescriptionSelection(
        blocks=selected,
        primary_field=primary_field,
        supplementary_fields=supplementary_fields,
        skipped_fields=skipped_fields,
        truncated=truncated,
        dropped_blocks=dropped,
        total_bytes=total_bytes,
        used_bytes=used,
    )


def select_description_source_from_payload(
    payload: Any,
    app_id: Optional[int] = None,
    *,
    max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
) -> DescriptionSelection:
    """Apply the source policy to a whole ``appdetails`` envelope."""

    return select_description_source(
        _extract_data(payload, app_id),
        max_source_bytes=max_source_bytes,
    )


def assess_source_sufficiency(
    selection: DescriptionSelection,
    *,
    min_source_tokens: int = DEFAULT_MIN_SOURCE_TOKENS,
) -> SourceSufficiency:
    """Assess whether the selected source can support the target length.

    The assessment is **deterministic** and depends only on the selected
    source and the source budget:

    * the selected, deduplicated source is measured in whitespace-
      separated tokens, and
    * the measurement is compared against ``min_source_tokens``.

    Two rules keep the assessment honest:

    * **A truncated source is never sufficient.** The blocks the budget
      dropped are precisely the material that would have carried the
      translation past the target, so a truncated source is reported as
      insufficient even when the blocks that did fit would have been
      enough on their own.

    * **The measurement is never taken from the model output.** A short
      translation does not prove that the source was short. This
      function never sees the model's text at all.

    The threshold is documented in
    :data:`app.config.DEFAULT_MIN_SOURCE_TOKENS`, and the assessment
    only ever selects a **warning**. It never rejects a translation.
    """

    if min_source_tokens < 1:
        raise ValueError("min_source_tokens must be positive")

    tokens = selection.source_token_count
    sufficient = bool(tokens >= min_source_tokens) and not selection.truncated

    return SourceSufficiency(
        sufficient=sufficient,
        source_tokens=tokens,
        min_source_tokens=min_source_tokens,
        truncated=selection.truncated,
    )


__all__ = [
    "BLOCK_KIND_HEADING",
    "BLOCK_KIND_LIST_ITEM",
    "BLOCK_KIND_PARAGRAPH",
    "ROLE_PRIMARY",
    "ROLE_SUPPLEMENTARY",
    "ROLE_SUPPORTING",
    "SOURCE_FIELD_ABOUT",
    "SOURCE_FIELD_DETAILED",
    "SOURCE_FIELD_SHORT",
    "DescriptionBlock",
    "DescriptionSelection",
    "SourceSufficiency",
    "assess_source_sufficiency",
    "normalize_block_text",
    "parse_description_html",
    "parse_description_text",
    "select_description_source",
    "select_description_source_from_payload",
]
