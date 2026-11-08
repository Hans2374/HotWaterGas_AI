"""Pure parsing of a Steam ``appdetails`` payload into Steam facts.

Everything in this module is a **pure function** of its input. There is
no network access, no clock access and no I/O, so every mapping rule can
be tested offline against a sanitized fixture.

Two rules shape the whole module:

**Only a defensible mapping is made.** A Steam field is mapped when it
has an unambiguous meaning in the description contract. Everything else — genres,
store categories, prices, packages, DLC, achievements, external URLs —
is dropped here, *before* the values can reach a prompt.

**The evidence is created here.** The parser is the owner of the
evidence it emits: it mints the ids, it writes the bounded excerpts and
it records the source reference. A model never authors this content.

HTML stripping
--------------
Steam serves requirement strings as small HTML fragments
(``<br>``, ``<strong>``, ``&nbsp;``). They are stripped to plain text
with a small, dependency-free converter. The converter is deliberately
conservative: it removes tags and decodes the handful of entities Steam
actually uses, and it does not attempt to be a general HTML parser.
"""

from __future__ import annotations

import html
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from app.schemas.description_fields import EvidenceReference
from app.retrieval.steam_models import (
    SteamFacts,
    SteamPlatformAvailability,
    SteamRequirements,
)


#: The single allow-listed Steam endpoint. There is no other.
STEAM_APPDETAILS_ENDPOINT = "https://store.steampowered.com/api/appdetails"

#: The fixed locale used for every request. Steam localises some fields
#: (notably ``release_date.date``), so the locale is pinned rather than
#: negotiated, and recorded in the evidence source reference so a stored
#: fact can be reproduced.
STEAM_LOCALE = "vietnamese"

#: The English locale.
#:
#: :data:`STEAM_LOCALE` is the default locale of the appdetails
#: request. The translation path retrieves the **English** store
#: description, which is then translated into Vietnamese by the model; it
#: therefore needs the English source text. The two locales are
#: deliberately separate names so a change to one can never alter the
#: other.
STEAM_LOCALE_ENGLISH = "english"

#: The fixed country used for every request, for the same reason.
STEAM_COUNTRY = "vn"

#: Maximum length of an evidence excerpt. The excerpt is a bounded
#: snippet that supports one specific fact, never a response body.
MAX_EXCERPT_CHARS = 500

#: Maximum length of a mapped Steam *value* (a requirement line, the
#: game title). The value and its evidence excerpt are bounded by the
#: same limit, so the excerpt always describes the whole value rather
#: than a prefix of it.
#:
#: The two must agree. If the value were longer than the excerpt, the
#: deterministic verification pass would compare a field value against
#: a *different* string than the one the evidence record carries, and
#: the comparison would silently reach the wrong verdict.
MAX_VALUE_CHARS = 500

#: The requirement fields the description contract models, in contract order.
_REQUIREMENT_FIELDS = ("os", "processor", "memory", "graphics", "storage")

#: Steam platform keys mapped to the canonical platform names used by
#: the description contract. Only truthy keys are mapped.
_PLATFORM_KEYS: Tuple[Tuple[str, str], ...] = (
    ("windows", "Windows"),
    ("mac", "macOS"),
    ("linux", "Linux"),
)

#: The Steam source path of each platform key.
#:
#: The key is the one Steam uses in its ``platforms`` object, and the
#: path is the full path of that key inside the ``appdetails`` payload,
#: so an evidence record can name exactly where its fact was read from.
_PLATFORM_SOURCE_PATHS: Dict[str, str] = {
    "windows": "platforms.windows",
    "mac": "platforms.mac",
    "linux": "platforms.linux",
}

#: The Steam source path of the release date.
_RELEASE_DATE_SOURCE_PATH = "release_date.date"

#: The Steam source path of the PC requirement tiers.
#:
#: Only the PC tier has a source path. Steam's ``mac_requirements`` and
#: ``linux_requirements`` are never mapped, so they never have one.
_PC_REQUIREMENTS_SOURCE_PATHS: Dict[str, str] = {
    "minimum": "pc_requirements.minimum",
    "recommended": "pc_requirements.recommended",
}
#: The labels Steam uses when it serves a requirement tier as one HTML
#: string instead of a structured object.
#:
#: Steam has two shapes for ``pc_requirements.minimum``: an object with
#: one key per requirement field, and a single HTML string that spells
#: the fields out as ``"<label>: value"`` lines. Both are handled, and
#: both are handled **deterministically** — a label is matched
#: literally, never guessed from context.
_REQUIREMENT_LABELS: Tuple[Tuple[str, str], ...] = (
    ("os", "Operating System"),
    ("processor", "Processor"),
    ("memory", "Memory"),
    ("graphics", "Graphics"),
    ("storage", "Storage"),
)

#: The labels above, lowercased and stripped, as they appear once the
#: HTML has been removed and the text has been casefolded.
_REQUIREMENT_LABEL_KEYS: Dict[str, str] = {
    label.casefold(): field_name for field_name, label in _REQUIREMENT_LABELS
}

#: Extra requirement labels a caller may opt into.
#:
#: Steam does not always spell a requirement label out. The Portal 2
#: store page writes its operating-system line as ``"OS *:"`` - the
#: asterisk marks a footnote - while the label this module matches by
#: default is ``"Operating System:"``.
#:
#: The extra labels are **opt-in** through the ``extra_labels``
#: parameter of :func:`map_appdetails`. The default is empty, so the
#: default path extracts exactly what it always extracted: a store page
#: that spells the label ``"OS *:"`` contributes nothing to the draft,
#: exactly as before. The translation path opts in, because a description
#: pipeline that cannot name the operating system would be reporting a
#: PC requirement set without its most important field.
_EXTRA_REQUIREMENT_LABELS: Dict[str, str] = {
    "os": "os",
    "os *": "os",
    "operating system *": "os",
}

#: Steam's own platform labels, used only to match a caller-supplied
#: platform string against a Steam platform. Steam calls macOS "Mac OS X".
_PLATFORM_ALIASES: Dict[str, str] = {
    "windows": "Windows",
    "win": "Windows",
    "win32": "Windows",
    "win64": "Windows",
    "pc": "Windows",
    "mac": "macOS",
    "macos": "macOS",
    "mac os x": "macOS",
    "os x": "macOS",
    "osx": "macOS",
    "linux": "Linux",
}

#: HTML tags removed from Steam requirement strings.
_TAG_RE = re.compile(r"<[^>]*>")

#: A line break rendered as whitespace.
_BR_RE = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)

#: Whitespace runs collapsed to a single space.
_WS_RE = re.compile(r"\s+")

#: The English month names Steam uses in ``release_date.date``.
#:
#: Steam localises this field, so both the English and the Vietnamese
#: spellings are accepted. The English month is used because the date
#: is emitted in ISO form and the locale only affects the *input*
#: spelling.
_ENGLISH_MONTHS: Dict[str, int] = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _build_month_table() -> Dict[str, int]:
    """Return the month lookup used by :func:`parse_steam_date`.

    The Vietnamese locale abbreviates the month to ``"Thg"`` and writes
    the month **number next to the name** - with or without a space:
    ``"13 Thg11, 2007"`` and ``"13 Thg 11, 2007"`` are the same date.
    :data:`_VIETNAMESE_MONTH_RE` folds both spellings into a single
    canonical token (``"thg11"``) before the lookup, so the table only
    ever needs the attached form.

    The full Vietnamese word (``"Thang 11"``) is folded the same way.
    """

    table = dict(_ENGLISH_MONTHS)
    for number in range(1, 13):
        table[f"thg{number}"] = number
        table[f"thang{number}"] = number
    return table


#: Every month spelling the date parser accepts, keyed by its canonical
#: (lowercase, space-free) token.
_MONTHS: Dict[str, int] = _build_month_table()

#: The Vietnamese month spelling, with the month number attached to or
#: separated from the name: ``"Thg11"``, ``"Thg 11"``, ``"Thang 11"``.
#:
#: The match is applied to the whole date string before the string is
#: split into tokens, so the month is recognised whether Steam put the
#: number against the name or separated it with a space.
_VIETNAMESE_MONTH_RE = re.compile(r"\b(thg|thang)\s*(\d{1,2})\b")

#: The lowest and highest year a Steam release date may carry. A year
#: outside the range is not a release date, and a date without a
#: four-digit year is ambiguous, so it is rejected rather than guessed.
MIN_STEAM_YEAR = 1000
MAX_STEAM_YEAR = 2999


#: Roman numerals used as sequel/edition markers in a game title.
#:
#: A title that ends with one of these tokens is a *different* product
#: from the same title without it: "Dark Souls" and "Dark Souls II" are
#: two games with different system requirements.
_ROMAN_NUMERALS = frozenset(
    {
        "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
        "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx",
    }
)

#: Tokens that mark a title as a *different edition* of the same game.
#:
#: These are matched case-insensitively as whole words. A title that
#: ends with one of them is not interchangeable with the bare title.
_EDITION_MARKERS = frozenset(
    {
        "edition",
        "editions",
        "definitive",
        "ultimate",
        "deluxe",
        "premium",
        "gold",
        "complete",
        "collection",
        "anniversary",
        "remastered",
        "remaster",
        "enhanced",
        "extended",
        "director",
        "directors",
        "goty",
        "goty edition",
        "game of the year edition",
    }
)

#: A numeric sequel/edition marker: "4", "II", "2nd".
#:
#: Only a *trailing* number is treated as a marker. A number in the
#: middle of a title ("Resident Evil 4 Village", "007") is part of the
#: name and carries no marker meaning.
#:
#: A trailing four-digit *year* is deliberately NOT covered by this
#: pattern. Steam appends a release year to disambiguate re-releases of
#: the same game ("Call of Duty 4: Modern Warfare (2007)"), and that
#: year never denotes a different product from the bare title. See
#: :data:`_YEAR_SUFFIX_RE` and :func:`_split_release_year`.
_NUMERIC_EDITION_RE = re.compile(r"^(?:\d+(?:st|nd|rd|th)?)$")

#: A trailing release-year suffix, e.g. "(2007)" or "[2007]".
#:
#: This is matched against the **raw** title, before normalisation,
#: because :func:`_normalize_title` deletes punctuation - including the
#: parentheses - so by the time a title is normalised the year is a bare
#: token ("... 2007") that is indistinguishable from a franchise index
#: ("Portal 2"). Stripping the year from the raw title is what keeps a
#: release year from being mistaken for a sequel marker.
#:
#: Only a *trailing* parenthesised year is a release-year suffix. A year
#: elsewhere in the title ("Fahrenheit 451") is part of the name.
_YEAR_SUFFIX_RE = re.compile(r"\s*[\(\[](\d{4})[\)\]]\s*$")

#: The plausible range of a release year. The lower bound excludes the
#: three-digit years that are franchise indices ("Portal 2", "Doom 3"),
#: and the upper bound excludes a day number that could never be a year.
MIN_RELEASE_YEAR = 1000
MAX_RELEASE_YEAR = 2999

# ── Text helpers ──────────────────────────────────────────────────────────


def strip_html(text: Any) -> Optional[str]:
    """Reduce a Steam HTML fragment to plain text.

    Returns ``None`` when the input is absent or carries no text at all,
    so a missing value is represented as a missing value rather than as
    an empty string.
    """

    if not isinstance(text, str):
        return None
    without_breaks = _BR_RE.sub(" ", text)
    without_tags = _TAG_RE.sub("", without_breaks)
    decoded = html.unescape(without_tags)
    collapsed = _WS_RE.sub(" ", decoded).strip()
    if not collapsed:
        return None
    return collapsed


def bound_value(text: Optional[str]) -> Optional[str]:
    """Return a Steam value, bounded to the contract's field limit.

    A Steam requirement line is external content of unknown length.
    The description contract caps a requirement value at
    :data:`MAX_VALUE_CHARS`, so a longer value is bounded here - at the
    boundary where external content enters the service - rather than
    being allowed to overflow the contract downstream.

    The value and its evidence excerpt are bounded by the same limit,
    so an excerpt always describes the whole value. That matters: the
    deterministic verification pass compares a field value against the
    excerpt of the record that claims to support it, and if the two
    were different strings the comparison would reach the wrong
    verdict on its own.

    **Truncation is not free.** A bounded value is only a *prefix* of
    the source fact, so the fact that truncation happened is recorded
    by :func:`bound_value_with_status` and travels with the value into
    the evidence record. This function keeps the plain-value signature
    for callers that do not need the status; the mapping code uses
    :func:`bound_value_with_status` so the flag is never lost.
    """

    bounded, _truncated = bound_value_with_status(text)
    return bounded


def bound_value_with_status(
    text: Optional[str],
) -> Tuple[Optional[str], bool]:
    """Return a bounded Steam value and whether it was truncated.

    The second element of the tuple is ``True`` when the source value
    was longer than :data:`MAX_VALUE_CHARS` and was therefore
    shortened.

    The flag exists because **value/excerpt agreement is not
    completeness**. When a value is truncated, the excerpt is cut from
    the already-shortened value, so the two are byte-identical and the
    deterministic comparison reaches ``SUPPORT``. That agreement only
    proves that the two agree *with each other* — it says nothing about
    whether the complete source fact was captured. Recording the
    truncation lets the verification pass distinguish "the source said
    exactly this" from "the source said more than this", and refuse to
    promote the latter.

    A value that is exactly at the limit is **not** truncated: it was
    not shortened, so it is a complete source fact.
    """

    if text is None:
        return None, False
    if len(text) <= MAX_VALUE_CHARS:
        return text, False
    return text[:MAX_VALUE_CHARS].rstrip(), True


def normalize_requirement_text(text: Optional[str]) -> Optional[str]:
    """Normalise a requirement string for conservative comparison.

    The comparison must treat ``"Windows 10  64-bit"`` and
    ``"windows 10 64-bit"`` as the same value while still treating
    ``"8 GB RAM"`` and ``"16 GB RAM"`` as different. The normalisation
    therefore only strips HTML, collapses whitespace and casefolds — it
    never removes digits, units or punctuation.
    """

    if text is None:
        return None
    stripped = strip_html(text)
    if stripped is None:
        return None
    return stripped.casefold()


def canonical_platform(name: str) -> Optional[str]:
    """Map a platform label to its canonical contract name."""

    key = _WS_RE.sub(" ", str(name).strip().casefold())
    return _PLATFORM_ALIASES.get(key)


def parse_steam_date(value: Any) -> Optional[str]:
    """Parse Steam's ``release_date.date`` into an ISO ``YYYY-MM-DD``.

    Steam returns a human-readable, locale-dependent string. The
    English locale spells it ``"Mar 25, 2022"``; the Vietnamese locale
    abbreviates the month and attaches the month number to it -
    ``"13 Thg11, 2007"`` or ``"13 Thg 11, 2007"``. Both are accepted.

    Anything that cannot be parsed **with confidence** returns ``None``
    so the field stays unknown rather than becoming a wrong date. In
    particular:

    * an unrecognised month, a missing day or a missing year is unknown,
    * a year outside :data:`MIN_STEAM_YEAR`..:data:`MAX_STEAM_YEAR` is
      unknown, because a Steam release date is never in the far past or
      the far future,
    * an impossible day for the month (``"31 Thg 2, 2022"``) is
      unknown, and
    * a **two-digit** year is unknown: ``"13 Thg11, 07"`` is genuinely
      ambiguous, so it is never resolved by guessing a century.
    """

    if not isinstance(value, str):
        return None
    text = _WS_RE.sub(" ", value).strip().casefold()
    if not text:
        return None

    # Steam uses "Coming soon" / "To be announced" for unreleased games.
    if any(
        marker in text
        for marker in ("coming soon", "to be announced", "tba", "chua phat hanh")
    ):
        return None

    # ISO input passes straight through.
    iso_candidate = value.strip()
    try:
        return datetime.strptime(iso_candidate, "%Y-%m-%d").date().isoformat()
    except ValueError:
        pass

    # The Vietnamese month spelling is folded into one canonical token
    # *before* the string is split, so "Thg11" and "Thg 11" are
    # recognised as the same month rather than as two unknown tokens.
    text = _fold_vietnamese_month(text)

    # "<day> <month> <year>" and "<month> <day>, <year>" — both
    # orderings, in either locale. The two branches differ only in
    # which token is the day, and each is tried against the month table
    # so a swapped spelling cannot produce a wrong date.
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        return None

    for day_token, month_token, year_token in (
        (parts[0], parts[1], parts[2]),
        (parts[1], parts[0], parts[2]),
    ):
        parsed = _compose_iso_date(day_token, month_token, year_token)
        if parsed is not None:
            return parsed

    return None


def _fold_vietnamese_month(text: str) -> str:
    """Fold the Vietnamese month spelling into one canonical token.

    ``"13 thg 11, 2007"`` becomes ``"13 thg11 2007"`` and
    ``"13 thg11, 2007"`` is left alone, so the token splitter always
    sees the month as a single token. The fold is a pure textual
    rewrite: it never reorders the parts of the date.
    """

    def _replace(match: "re.Match[str]") -> str:
        return f"{match.group(1)}{match.group(2)}"

    return _VIETNAMESE_MONTH_RE.sub(_replace, text)


def _compose_iso_date(
    day_token: str, month_token: str, year_token: str
) -> Optional[str]:
    """Return the ISO date for one day/month/year triple, or ``None``.

    The triple is only accepted when every part is unambiguous: the
    month must be a known month, the day and the year must be plain
    numbers, and the year must be a four-digit year inside the
    plausible range.
    """

    month_number = _MONTHS.get(month_token)
    if month_number is None:
        return None
    if not day_token.isdigit() or not year_token.isdigit():
        return None

    year = int(year_token)
    if len(year_token) != 4 or not MIN_STEAM_YEAR <= year <= MAX_STEAM_YEAR:
        return None

    day = int(day_token)
    if not 1 <= day <= 31:
        return None

    try:
        return datetime(year, month_number, day).date().isoformat()
    except ValueError:
        # An impossible day for that month ("31 Thg 2, 2022").
        return None


def titles_match(requested: str, retrieved: Optional[str]) -> Tuple[bool, str]:
    """Compare a caller-supplied product name with a Steam title.

    The policy is deliberately **conservative**: it accepts a title pair
    only when the two names are the *same* title, and it fails closed
    on everything else.

    The accepted case is a **normalised exact match**:

    * letter case is ignored,
    * punctuation is ignored (``"Aurora's Echo"`` == ``"AURORAS ECHO"``),
    * whitespace runs are collapsed,
    * a **trailing release year** is ignored on either side: Steam
      appends the year to disambiguate re-releases of the same game, so
      ``"Call of Duty 4: Modern Warfare"`` matches
      ``"Call of Duty 4: Modern Warfare (2007)"``.

    Everything else is a mismatch, in particular:

    * **one title being a substring of the other** is never enough —
      ``"Dark Souls"`` is not ``"Dark Souls II"``, and ``"Star Wars"``
      is not ``"Star Warship"``;
    * a **trailing sequel or edition marker** makes the titles
      different: an Arabic numeral (``"Resident Evil 4"``), a Roman
      numeral (``"StarCraft II"``) or an edition word (``"Elden Ring
      Definitive Edition"``) all denote a distinct product;
    * a **subtitle** (``"Portal 2"`` vs ``"Portal 2 Co-op"``) makes the
      titles different;
    * **two different release years** make the titles different
      (``"Doom (1999)"`` vs ``"Doom (2016)"``), because they denote
      different releases of the same name.

    A mismatch is never resolved by picking the closest title, and no
    external name search is performed: the whole point is that a wrong
    App ID must not silently enrich a product. The service turns a
    mismatch into the existing identity-mismatch contract, so no Steam
    fact reaches the draft.

    The release-year handling is deliberately narrow. Only a **trailing
    parenthesised four-digit year** in the range 1000-2999 is treated as
    a release-year suffix, and it is extracted from the **raw** title
    before normalisation - otherwise the year would be indistinguishable
    from the franchise index in ``"Portal 2"``. A franchise index is
    never stripped, so every sequel and edition rejection above is
    unaffected. The year is removed **for the comparison only**: the
    retrieved title that reaches the facts and the evidence record is
    the untouched Steam title, year included.

    Known limitation
    ----------------
    String normalisation cannot decide game identity in general. Two
    genuinely different products can share a normalised title, and a
    legitimately renamed product will not match its former title. This
    policy resolves both cases the same way: it reports a mismatch and
    lets the caller re-assert the App ID explicitly. A **false
    negative** (a correct App ID refused) is recoverable; a **false
    positive** (another game's requirements attached to this product)
    is not.
    """

    if retrieved is None:
        return False, "steam_reported_no_title"

    # A trailing release year is a disambiguation aid, not a sequel or
    # edition marker. It is stripped from BOTH titles before the
    # comparison, so that "Call of Duty 4: Modern Warfare" and Steam's
    # "Call of Duty 4: Modern Warfare (2007)" are recognised as the
    # same game. The split happens on the raw titles, because the
    # normalisation below deletes the parentheses and would leave the
    # year indistinguishable from a franchise index.
    requested_base, requested_year = _split_release_year(requested)
    retrieved_base, retrieved_year = _split_release_year(retrieved)

    normalized_requested = _normalize_title(requested_base)
    normalized_retrieved = _normalize_title(retrieved_base)
    if not normalized_requested:
        return False, "requested_name_is_empty"
    if not normalized_retrieved:
        return False, "steam_title_is_empty"

    # Both titles carry a release year: the years must agree, or the
    # titles name different products ("Doom (1999)" vs "Doom (2016)").
    if (
        requested_year is not None
        and retrieved_year is not None
        and requested_year != retrieved_year
    ):
        return False, "release_year_differs"

    if normalized_requested == normalized_retrieved:
        return True, "exact_match"

    requested_marker = _marker_kind(normalized_requested)
    retrieved_marker = _marker_kind(normalized_retrieved)

    # A trailing sequel/edition marker on exactly one of the two titles
    # means they are different products — but only when the two titles
    # are actually related. "Dark Souls" vs "Dark Souls II" is a
    # sequel; "ELDEN RING" vs "Half-Life 2" is not.
    if (requested_marker is None) != (retrieved_marker is None):
        marker_title = (
            normalized_requested
            if requested_marker is not None
            else normalized_retrieved
        )
        other_title = (
            normalized_retrieved
            if requested_marker is not None
            else normalized_requested
        )
        if _shares_base(marker_title, other_title):
            return False, "sequel_or_edition_marker"

    # Both titles carry a marker: "Diablo II" vs "Diablo III".
    elif requested_marker is not None and retrieved_marker is not None:
        if _shares_base(
            _strip_marker(normalized_requested),
            _strip_marker(normalized_retrieved),
        ):
            return False, "sequel_or_edition_marker"

    # One title is the other plus extra words: a subtitle or an
    # extended edition ("Elden Ring" vs "Elden Ring Shadow of the
    # Erdtree"). Different products.
    if _is_token_prefix(normalized_requested, normalized_retrieved):
        return False, "title_has_subtitle"

    # One title continues the other mid-word: "Star Wars" vs
    # "Star Warship". Also a different product. The earlier defect
    # accepted this pair because the shorter title was long enough.
    if (
        normalized_requested.startswith(normalized_retrieved)
        or normalized_retrieved.startswith(normalized_requested)
    ):
        return False, "title_is_prefix_of_other"

    return False, "title_differs"


def _strip_marker(normalized: str) -> str:
    """Return the title without its trailing marker token."""

    tokens = normalized.split()
    if len(tokens) <= 1:
        return ""
    return " ".join(tokens[:-1])


def _shares_base(first: str, second: str) -> bool:
    """Return ``True`` when two titles share a common base name.

    "Dark Souls" and "Dark Souls II" share the base "Dark Souls";
    "ELDEN RING" and "Half-Life 2" share nothing. The comparison is
    deliberately generous about the *base* — a base that extends the
    other title is accepted, so "Elden Ring" and "Elden Ring
    Definitive Edition" are recognised as related — but it never
    relates two titles that have nothing in common.
    """

    if not first or not second:
        return False
    if first == second:
        return True
    if first.startswith(second) or second.startswith(first):
        return True
    return False


def _is_token_prefix(first: str, second: str) -> bool:
    """Return ``True`` when one title is the other plus whole words.

    This is the *word* boundary: "elden ring" is a token-prefix of
    "elden ring shadow of the erdtree" but not of "elden ring ship",
    because "ring" and "ship" are different words.
    """

    first_tokens = first.split()
    second_tokens = second.split()
    if len(first_tokens) == len(second_tokens):
        return False
    shorter, longer = sorted((first_tokens, second_tokens), key=len)
    return longer[: len(shorter)] == shorter


def _marker_kind(normalized: str) -> Optional[str]:
    """Classify the trailing sequel/edition marker of a title.

    Returns ``"numeric"``, ``"roman"``, ``"edition"`` or ``None`` when
    the title carries no trailing marker. The title is already
    normalised, so it is a casefolded, punctuation free,
    whitespace-collapsed string.

    Only a *trailing* token is considered: a marker is a suffix
    (``"Diablo II"``), never a word that happens to appear somewhere
    in the middle of the title.
    """

    tokens = normalized.split()
    if not tokens:
        return None

    last = tokens[-1]

    # A trailing numeral: "Resident Evil 4", "Half-Life 2".
    if _NUMERIC_EDITION_RE.match(last):
        return "numeric"

    # A trailing Roman numeral: "Dark Souls II", "StarCraft III".
    # A single "I" is included: "Portal I" and "Portal" are different
    # products just as "Portal II" and "Portal" are.
    if last in _ROMAN_NUMERALS:
        return "roman"

    # A trailing edition word: "Elden Ring Definitive Edition".
    if last in _EDITION_MARKERS:
        return "edition"

    return None


def _split_release_year(raw: str) -> Tuple[str, Optional[int]]:
    """Split a trailing release-year suffix off a **raw** title.

    Returns ``(title_without_year, year_or_None)``.

    Steam appends a release year to disambiguate re-releases of the same
    game: the store title of the 2007 *Call of Duty 4: Modern Warfare* is
    ``"Call of Duty 4: Modern Warfare (2007)"``. The year is a
    **disambiguation aid, not a sequel marker** - it never denotes a
    different product from the bare title. Treating it as a marker made
    the identity check reject a correct App ID.

    The split is done on the **raw** title, before normalisation, because
    :func:`_normalize_title` deletes the parentheses. Once normalised,
    the year is a bare ``"2007"`` token that cannot be told apart from
    the franchise index in ``"Portal 2"`` - which *is* a genuine sequel
    marker. Splitting the raw title is what keeps the two apart.

    The year is only removed when:

    * it is a trailing parenthesised four-digit number,
    * it falls inside the plausible release-year range, and
    * a non-empty base title remains afterwards.

    A year anywhere other than the trailing position ("Fahrenheit 451")
    is part of the name and is left alone.
    """

    if not isinstance(raw, str):
        return raw, None

    match = _YEAR_SUFFIX_RE.search(raw)
    if not match:
        return raw, None

    year = int(match.group(1))
    if not MIN_RELEASE_YEAR <= year <= MAX_RELEASE_YEAR:
        return raw, None

    base = raw[: match.start()].strip()
    if not base:
        # The title is nothing but a year: there is no base to compare.
        return raw, None

    return base, year


def _normalize_title(text: str) -> str:
    """Strip a title down to comparable characters.

    Case is folded, punctuation is dropped and whitespace runs are
    collapsed. Digits and letters are kept: they are what separates
    "Half-Life" from "Half-Life 2", so removing them would recreate
    the very conflation this module exists to prevent.

    A **hyphen** is treated as a word separator rather than simply
    deleted, because a hyphen joins two words that stay separate words:
    "Elden-Ring" and "Elden Ring" are the same title, while deleting
    the hyphen outright would make them "eldenring" and "elden ring".
    An **apostrophe**, by contrast, is deleted: it marks possession or
    a contraction, so "Aurora's Echo" and "AURORAS ECHO" are the same
    title.
    """

    lowered = str(text).strip().casefold()
    # A hyphen separates words; every other punctuation mark is dropped.
    spaced = lowered.replace("-", " ")
    kept = [ch for ch in spaced if ch.isalnum() or ch.isspace()]
    collapsed = _WS_RE.sub(" ", "".join(kept)).strip()
    return collapsed


# ── Evidence construction ─────────────────────────────────────────────────


def make_evidence_id(app_id: int, field_path: str) -> str:
    """Mint a service-owned, collision-free evidence id.

    The id embeds the App ID and the field path so it is self-describing
    in a log and impossible for a model to guess by accident.
    """

    safe_path = field_path.replace(".", "_")
    return f"steam:{app_id}:{safe_path}"


def make_source_ref(app_id: int, *, locale: Optional[str] = None) -> str:
    """Return the reproducible source reference for one retrieval.

    ``locale`` names the Steam locale the request was made with. It
    defaults to the default locale, so the evidence records are unchanged.
    The translation path passes the English locale, so its records name the
    locale that was actually requested.
    """

    effective_locale = locale or STEAM_LOCALE
    return (
        f"{STEAM_APPDETAILS_ENDPOINT}?appids={app_id}"
        f"&l={effective_locale}&cc={STEAM_COUNTRY}"
    )


def build_evidence(
    app_id: int,
    field_path: str,
    excerpt: str,
    retrieved_at: str,
    *,
    truncated: bool = False,
    source_field: Optional[str] = None,
    locale: Optional[str] = None,
) -> EvidenceReference:
    """Create one service-owned evidence record.

    ``truncated`` records that the excerpt is only a **prefix** of the
    source fact, because the source value was longer than the
    contract's field limit. The flag is set here — at the boundary
    where the external content is bounded — and it travels with the
    record into the response, so the verification pass can tell a
    complete fact from a shortened one. A model can never set it: the
    service replaces every surviving evidence entry with its own
    record.

    ``source_field`` names the field **inside the Steam payload** the
    record was read from — ``"release_date.date"``, ``"platforms"``,
    ``"pc_requirements.minimum"``. It is the provenance of the *source*,
    as opposed to ``field_path``, which names the *draft* field the
    record supports. Like every other property of the record, it is
    service-owned: the service replaces any model-authored version of
    the record with its own.
    """

    return EvidenceReference(
        evidence_id=make_evidence_id(app_id, field_path),
        field_path=field_path,
        source_type="steam",
        source_ref=make_source_ref(app_id, locale=locale),
        retrieved_at=retrieved_at,
        excerpt=excerpt[:MAX_EXCERPT_CHARS],
        truncated=truncated,
        source_field=source_field,
    )


# ── The mapper ────────────────────────────────────────────────────────────


def _extract_data(payload: Any, app_id: int) -> Optional[Dict[str, Any]]:
    """Return the ``data`` object for one App ID.

    The Steam envelope is ``{"<app_id>": {"success": true, "data": {...}}}``.
    The App ID key is looked up first. As a fallback, a payload that
    holds exactly one entry with a ``data`` object is accepted — but
    only when it is unambiguous. A payload that is missing its ``data``
    object entirely returns ``None`` so the caller reports "not found"
    instead of mapping an empty document.
    """

    if not isinstance(payload, dict):
        return None

    entry = payload.get(str(app_id))
    if not isinstance(entry, dict):
        candidates = [
            value
            for value in payload.values()
            if isinstance(value, dict) and isinstance(value.get("data"), dict)
        ]
        if len(candidates) != 1:
            return None
        entry = candidates[0]

    data = entry.get("data")
    if not isinstance(data, dict):
        return None
    return data


def map_appdetails(
    app_id: int,
    payload: Any,
    *,
    retrieved_at: str,
    locale: Optional[str] = None,
    extra_requirement_labels: bool = False,
) -> Tuple[SteamFacts, List[EvidenceReference], Dict[str, str]]:
    """Map a Steam ``appdetails`` payload into facts and evidence.

    Returns ``(facts, evidence, evidence_by_field)``. ``evidence_by_field``
    maps a dotted field path to the evidence id that supports it, so
    the merge step can attach provenance without re-deriving it.

    Fields that Steam did not provide are simply absent from the result:
    no placeholder is invented and nothing is copied between tiers.

    ``locale`` names the Steam locale the payload was fetched with. It
    only affects the recorded source reference; the mapping itself is
    identical for every locale. It defaults to the default locale, so
    the default behaviour is unchanged.

    ``extra_requirement_labels`` opts into the additional requirement
    spellings Steam uses (``"OS *:"``). It defaults to ``False``, so the
    default path extracts exactly what it always extracted.
    """

    data = _extract_data(payload, app_id)
    if data is None:
        return SteamFacts(app_id=app_id), [], {}

    evidence: List[EvidenceReference] = []
    evidence_by_field: Dict[str, str] = {}

    def _record(
        field_path: str,
        excerpt: Optional[str],
        *,
        truncated: bool = False,
        source_field: Optional[str] = None,
    ) -> None:
        if not excerpt:
            return
        item = build_evidence(
            app_id,
            field_path,
            excerpt,
            retrieved_at,
            truncated=truncated,
            source_field=source_field,
            locale=locale,
        )
        evidence.append(item)
        evidence_by_field[field_path] = item.evidence_id

    # ── name ─────────────────────────────────────────────────────────
    raw_name = data.get("name")
    name = strip_html(raw_name) if isinstance(raw_name, str) else None
    # The name is bounded like any other external value, and the fact
    # that it was shortened is recorded on the evidence record.
    name, name_truncated = bound_value_with_status(name)
    if name:
        _record("name", name, truncated=name_truncated, source_field="name")

    # ── release date ─────────────────────────────────────────────────
    release_date = None
    release_container = data.get("release_date")
    if isinstance(release_container, dict):
        release_date = parse_steam_date(release_container.get("date"))
    if release_date:
        _record(
            "release_date",
            release_date,
            source_field=_RELEASE_DATE_SOURCE_PATH,
        )

    # ── platforms ────────────────────────────────────────────────────
    platforms, availability = _map_platforms(data.get("platforms"))
    if platforms:
        # The draft models the platform list as one field, so the record
        # is bound to "platforms". Its source names the **individual**
        # Steam keys the list was built from, so each platform
        # statement is traceable to the key it came from.
        source_fields = [
            _PLATFORM_SOURCE_PATHS[key]
            for key, _label in _PLATFORM_KEYS
            if _availability_for(availability, key) is True
        ]
        _record(
            "platforms",
            ", ".join(platforms),
            source_field=", ".join(source_fields),
        )

    # ── PC requirements ──────────────────────────────────────────────
    minimum, recommended = _map_requirements(
        data,
        evidence,
        evidence_by_field,
        app_id,
        retrieved_at,
        locale=locale,
        extra_requirement_labels=extra_requirement_labels,
    )

    facts = SteamFacts(
        app_id=app_id,
        name=name,
        release_date=release_date,
        platforms=platforms,
        platform_availability=availability,
        minimum=minimum,
        recommended=recommended,
    )
    return facts, evidence, evidence_by_field


def _map_platforms(
    platform_container: Any,
) -> Tuple[List[str], SteamPlatformAvailability]:
    """Map Steam's ``platforms`` object into the two platform views.

    Two things are extracted, and they are deliberately kept apart:

    * ``platforms`` — the canonical names of the platforms Steam
      **positively reported as supported**. This is the list the
      contract models, and it is unchanged in meaning: a platform Steam
      reported as unsupported simply does not appear in it.

    * ``availability`` — the three-state view: supported, explicitly
      unsupported, or not reported at all. A platform Steam reported as
      ``false`` and a platform Steam did not mention are **different
      facts**, and the supported-only list cannot tell them apart.

    The distinction is preserved deliberately:

    * a missing key, a ``null`` value and a value of an unexpected
      type are all **not reported** (``None``) — never "unsupported",
    * only a literal ``True`` means supported, and only a literal
      ``False`` means explicitly unsupported, so a truthy non-boolean
      (``"true"``, ``1``) is not accepted as Steam saying yes, and
    * platform support is never inferred from the system requirements:
      a game that lists macOS requirements is not thereby recorded as
      a macOS product.

    The mapping is a pure function of the payload, and it never
    inspects ``pc_requirements``, ``mac_requirements`` or
    ``linux_requirements``.
    """

    supported: List[str] = []
    availability = SteamPlatformAvailability()

    if not isinstance(platform_container, dict):
        return supported, availability

    for key, label in _PLATFORM_KEYS:
        raw = platform_container.get(key)

        # Only a literal boolean is a platform statement. Anything else
        # — including a truthy string or a number — is treated as "not
        # reported", because Steam's contract for this object is a
        # boolean and a non-boolean cannot be interpreted as one.
        if raw is True:
            availability = _replace(availability, key, True)
            supported.append(label)
        elif raw is False:
            availability = _replace(availability, key, False)
        # else: not reported. The value stays None.

    return supported, availability


def _replace(
    availability: SteamPlatformAvailability,
    key: str,
    value: bool,
) -> SteamPlatformAvailability:
    """Return ``availability`` with one platform's three-state value set."""

    if key == "windows":
        return SteamPlatformAvailability(windows=value, mac=availability.mac, linux=availability.linux)
    if key == "mac":
        return SteamPlatformAvailability(windows=availability.windows, mac=value, linux=availability.linux)
    if key == "linux":
        return SteamPlatformAvailability(windows=availability.windows, mac=availability.mac, linux=value)
    return availability


def _availability_for(
    availability: SteamPlatformAvailability,
    key: str,
) -> Optional[bool]:
    """Return the three-state value of one platform key."""

    if key == "windows":
        return availability.windows
    if key == "mac":
        return availability.mac
    if key == "linux":
        return availability.linux
    return None


def _map_requirements(
    data: Dict[str, Any],
    evidence: List[EvidenceReference],
    evidence_by_field: Dict[str, str],
    app_id: int,
    retrieved_at: str,
    *,
    locale: Optional[str] = None,
    extra_requirement_labels: bool = False,
) -> Tuple[SteamRequirements, SteamRequirements]:
    """Map ``pc_requirements`` into the two requirement tiers.

    Only the PC tier is mapped. Steam's ``mac_requirements`` and
    ``linux_requirements`` are **not** merged into the shared PC fields:
    the contract models one tier per field, not one tier per
    operating system, so folding the macOS values into
    ``system_requirements.minimum.os`` would silently attribute
    macOS-only requirements to a Windows product. That limitation is
    reported rather than papered over.
    """

    pc_requirements = data.get("pc_requirements")
    if not isinstance(pc_requirements, dict):
        return SteamRequirements(), SteamRequirements()

    minimum = _map_tier(
        pc_requirements.get("minimum"), "minimum", evidence, evidence_by_field, app_id, retrieved_at,
        locale=locale,
        extra_requirement_labels=extra_requirement_labels,
    )
    recommended = _map_tier(
        pc_requirements.get("recommended"),
        "recommended",
        evidence,
        evidence_by_field,
        app_id,
        retrieved_at,
        locale=locale,
        extra_requirement_labels=extra_requirement_labels,
    )
    return minimum, recommended


def _map_tier(
    tier_data: Any,
    tier_name: str,
    evidence: List[EvidenceReference],
    evidence_by_field: Dict[str, str],
    app_id: int,
    retrieved_at: str,
    *,
    locale: Optional[str] = None,
    extra_requirement_labels: bool = False,
) -> SteamRequirements:
    """Map one requirement tier into a :class:`SteamRequirements`.

    Two source shapes are accepted, because Steam uses both:

    * a **structured object** with one key per requirement field
      (``{"os": "Windows 10", "memory": "8 GB RAM"}``), and
    * a **single HTML string** that spells the fields out as
      ``"Operating System: Windows 10<br>Memory: 8 GB RAM"``.

    The string form is parsed **deterministically**: the HTML is
    removed, the text is split into lines, and each line is matched
    against the exact set of labels Steam uses. A line whose label is
    not one of them is ignored rather than guessed at, so a value is
    never attributed to the wrong requirement field.

    A tier is only ever populated from its own source key. The
    recommended tier is never derived from the minimum tier, and a
    missing field is left missing rather than filled in from a
    sibling.
    """

    values: Dict[str, Optional[str]] = {name: None for name in _REQUIREMENT_FIELDS}

    if isinstance(tier_data, dict):
        for field_name in _REQUIREMENT_FIELDS:
            raw = tier_data.get(field_name)
            text = strip_html(raw) if isinstance(raw, str) else None
            values[field_name] = text
    elif isinstance(tier_data, str):
        values.update(
            _parse_requirement_string(tier_data, extra_requirement_labels=extra_requirement_labels)
        )

    bounded: Dict[str, Optional[str]] = {}
    for field_name in _REQUIREMENT_FIELDS:
        # The value is bounded before the evidence excerpt is cut from
        # it, so the excerpt always describes the whole value. The
        # truncation flag travels with the record, so a value that was
        # shortened can never be promoted to 'verified' later on.
        text, truncated = bound_value_with_status(values.get(field_name))
        bounded[field_name] = text
        if text:
            path = f"system_requirements.{tier_name}.{field_name}"
            item = build_evidence(
                app_id,
                path,
                text,
                retrieved_at,
                truncated=truncated,
                source_field=_PC_REQUIREMENTS_SOURCE_PATHS.get(tier_name),
                locale=locale,
            )
            evidence.append(item)
            evidence_by_field[path] = item.evidence_id

    return SteamRequirements(
        os=bounded.get("os"),
        processor=bounded.get("processor"),
        memory=bounded.get("memory"),
        graphics=bounded.get("graphics"),
        storage=bounded.get("storage"),
    )


def _parse_requirement_string(
    text: str,
    *,
    extra_requirement_labels: bool = False,
) -> Dict[str, Optional[str]]:
    """Parse Steam's HTML-string form of one requirement tier.

    The result maps only the requirement fields the text actually
    names. A field the text does not mention is absent from the
    result, so it stays missing rather than being given an empty or
    invented value.

    The parse is deliberately literal:

    * the text is split into lines **before** the HTML is removed, so
      a ``<br>``-separated line is not merged with the line after it,
    * a line is split at its **first** colon, and
    * the label before the colon must match one of the labels Steam
      uses — case-insensitively, and ignoring surrounding whitespace.

    A line without a colon, or with an unrecognised label, is ignored.
    Nothing is inferred from the value that follows a known label, so
    a value can never be filed under a field it does not belong to.

    ``extra_requirement_labels`` opts into the additional spellings
    Steam uses (``"OS *:"``). It defaults to ``False``, so the
    recognises exactly the labels it always recognised.
    """

    parsed: Dict[str, Optional[str]] = {}
    label_keys = dict(_REQUIREMENT_LABEL_KEYS)
    if extra_requirement_labels:
        label_keys.update(_EXTRA_REQUIREMENT_LABELS)

    for raw_line in _split_requirement_lines(text):
        line = strip_html(raw_line)
        if line is None or ":" not in line:
            continue
        label, _, value = line.partition(":")
        field_name = label_keys.get(label.strip().casefold())
        if field_name is None:
            continue
        value = value.strip()
        if not value:
            continue
        # The first occurrence wins, so a repeated label does not
        # overwrite a value that was already read.
        parsed.setdefault(field_name, value)
    return parsed


def _split_requirement_lines(text: str) -> List[str]:
    """Split a requirement string into its individual lines.

    Steam separates the lines of its requirement string with ``<br>``
    tags or with real newlines. Both are honoured, and the split
    happens **before** the HTML is stripped — otherwise every
    ``<br>`` would collapse into a space and the whole tier would
    arrive as one long line.
    """

    without_breaks = _BR_RE.sub("\n", text)
    return [line for line in without_breaks.splitlines() if line.strip()]


__all__ = [
    "MAX_EXCERPT_CHARS",
    "MAX_VALUE_CHARS",
    "STEAM_APPDETAILS_ENDPOINT",
    "STEAM_COUNTRY",
    "STEAM_LOCALE",
    "STEAM_LOCALE_ENGLISH",
    "bound_value",
    "bound_value_with_status",
    "build_evidence",
    "canonical_platform",
    "make_evidence_id",
    "make_source_ref",
    "map_appdetails",
    "normalize_requirement_text",
    "parse_steam_date",
    "strip_html",
    "titles_match",
]
