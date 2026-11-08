"""Shared description value objects and normalisation helpers.

This module is **version-neutral**. It holds the value objects that the
description contracts have in common — the evidence record, the field
verification state, the release date, the platform list and the Windows
system-requirement tiers — together with the normalisation helpers that
turn "the model said it does not know" into an explicit ``null``.

Why the module exists
---------------------
The objects here were originally defined inside the versioned schema
modules. The retained pipeline needed them, so importing them from
a version-specific module tied the surviving code to code that is
scheduled for removal. Moving them here removes that coupling: the
pipeline now depends on a module whose name says nothing about which
contract version introduced it.

The module is deliberately **dependency-free**. It imports nothing from
this service, so it can be imported by schemas, services and retrieval
code without creating a cycle, and so the rules it encodes stay in one
place: a disagreement between the contract and the verification pass
about what "unknown" means would silently let an unknown sentinel pass
as a fact.

Schema identity
---------------
The JSON Schema documents derived from these models are sent verbatim to
the model provider and are also published through the OpenAPI document.
Both documents name their definitions after the model's class name.

These models are **version-neutral**, and their class identities say so:
``FieldVerification`` is called ``FieldVerification`` in the Python
source, in the generated JSON Schema, and in the OpenAPI document. The
symbol a caller imports and the name the provider sees are the same
name, so there is nothing to keep in sync and nothing to translate.

The *structure* of the generated document is the part that matters, and
it is protected by a test rather than by a naming convention. See
``tests/test_description_pipeline.py`` for the structural-invariance
guard that pins the provider-facing document's required set, property
set, enumerations, bounds and reference graph.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class ErrorResponse(BaseModel):
    """Stable error envelope.

    Every non-2xx response from the generation endpoint uses this shape so
    downstream callers can rely on a consistent contract. ``message`` is
    generic and intentionally non-revealing.
    """

    code: str
    message: str


# ── Vocabularies ─────────────────────────────────────────────────────────

#: Verification state of a single field.
VerificationStatus = Literal["verified", "unverified", "conflicting", "unknown"]

#: Where an evidence item came from.
EvidenceSourceType = Literal["caller_supplied", "steam", "manual", "unknown"]

#: The six requirement fields modelled independently for each tier.
REQUIREMENT_FIELD_NAMES: Tuple[str, ...] = (
    "os",
    "processor",
    "memory",
    "graphics",
    "storage",
    "notes",
)

#: The requirement fields that must be known for a draft to be "complete".
#: ``notes`` is deliberately excluded: a note is inherently optional.
COMPLETENESS_REQUIREMENT_FIELDS: Tuple[str, ...] = (
    "os",
    "processor",
    "memory",
    "graphics",
    "storage",
)


# ── Normalisation ────────────────────────────────────────────────────────

#: Values a model may use to mean "I do not know". They are normalised
#: to ``None`` so the payload never carries a string that pretends to
#: be a fact.
UNKNOWN_SENTINELS = frozenset(
    {
        "",
        "-",
        "--",
        "n/a",
        "na",
        "none",
        "null",
        "nil",
        "unknown",
        "unk",
        "unspecified",
        "not specified",
        "not stated",
        "not known",
        "not available",
        "not provided",
        "not applicable",
        "tbd",
        "tbd.",
        "chưa biết",
        "chua biet",
        "không rõ",
        "khong ro",
        "không xác định",
        "khong xac dinh",
        "chưa có",
        "chua co",
    }
)


def normalize_optional_text(value: Optional[str]) -> Optional[str]:
    """Return a trimmed string, or ``None`` for empty/unknown sentinels.

    The function is the single place where "the model said it does not
    know" is translated into the contract's explicit ``null``.
    """

    if value is None:
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.casefold() in UNKNOWN_SENTINELS:
        return None
    return stripped


def normalize_iso_date(value: Optional[str]) -> Optional[str]:
    """Normalise an ISO-8601 date/datetime string to ``YYYY-MM-DD``.

    Returns ``None`` when the value is empty, an unknown sentinel, or
    cannot be parsed as an ISO-8601 date. A full timestamp is truncated
    to its date part because the contract models a *date*, not an
    instant.
    """

    from datetime import date

    cleaned = normalize_optional_text(value)
    if cleaned is None:
        return None
    try:
        return date.fromisoformat(cleaned).isoformat()
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(cleaned.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None


def is_utc_iso_timestamp(value: str) -> bool:
    """Return ``True`` when ``value`` is an ISO-8601 instant in UTC.

    The check is strict on purpose. A verification timestamp must name
    an exact instant and say which timezone it is in, so the value must
    parse as a datetime and carry a UTC offset. A naive local time, a
    date without a time, or free text is rejected: a consumer reading
    the timestamp must be able to trust it.
    """

    text = str(value).strip()
    if not text:
        return False
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return False
    offset = parsed.utcoffset()
    if offset is None:
        # A naive datetime carries no timezone, so the instant is
        # ambiguous.
        return False
    return offset == timedelta(0)


# ── Building blocks ──────────────────────────────────────────────────────


class FieldVerification(BaseModel):
    """Verification state of a single field.

    ``status`` defaults to ``"unknown"`` so a field that omits the
    object entirely is treated as *nothing known* rather than *verified*.
    """

    model_config = ConfigDict(extra="forbid")

    status: VerificationStatus = Field(
        default="unknown",
        description=(
            "'verified' (checked against evidence), 'unverified' (a value "
            "exists but was never checked), 'conflicting' (sources "
            "disagree, no winner selected) or 'unknown' (nothing known)."
        ),
    )
    note: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Free-form note about the verification state. Never a fact.",
    )
    checked_at: Optional[str] = Field(
        default=None,
        max_length=64,
        description=(
            "ISO-8601 UTC timestamp of the last verification attempt, with "
            "an explicit UTC indicator (e.g. '2026-10-27T08:15:00+00:00'). "
            "Set only by the service, and only when a deterministic check "
            "actually ran: the presence of a timestamp is not proof that "
            "the check succeeded."
        ),
    )

    @field_validator("checked_at", mode="before")
    @classmethod
    def _validate_checked_at(cls, value: Any) -> Optional[str]:
        """Reject a malformed verification timestamp.

        A timestamp is only meaningful if it says *when*, in a known
        timezone. A free-text value would let a model (or a careless
        caller) claim a verification happened at "yesterday". The value
        must therefore be a real ISO-8601 instant carrying an explicit
        UTC offset.
        """

        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("checked_at must be a string or null")
        text = value.strip()
        if not text:
            return None
        if not is_utc_iso_timestamp(text):
            raise ValueError(
                "checked_at must be an ISO-8601 timestamp with an explicit "
                "UTC indicator, e.g. '2026-10-27T08:15:00+00:00'"
            )
        return text


class EvidenceReference(BaseModel):
    """A provenance entry. The id is referenced by fields via ``evidence_ids``."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(..., min_length=1, max_length=200)
    field_path: Optional[str] = Field(
        default=None,
        max_length=300,
        description=(
            "The dotted draft field path this record supports, e.g. "
            "'release_date' or 'system_requirements.minimum.os'. The "
            "binding is **server-owned**: the service sets it when it "
            "mints the record, and a model may never author or change "
            "it. A record that is not bound to the field it is cited "
            "for cannot justify a verified value. Absent on a record "
            "whose origin the service cannot establish."
        ),
    )
    source_type: EvidenceSourceType = Field(
        default="unknown",
        description=(
            "Where the evidence came from. 'caller_supplied' marks a fact "
            "the caller sent in the request; 'steam' marks a fact the "
            "service retrieved from the Steam store; 'manual' marks a fact "
            "entered by an administrator; 'unknown' marks a fact whose "
            "origin is not established. Only the service may set this "
            "field on a record that survives into a response: a model may "
            "reference an evidence id, but it may never author or vouch "
            "for a record."
        ),
    )
    source_ref: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Locator of the evidence (URL, app id, manual reference).",
    )
    source_field: Optional[str] = Field(
        default=None,
        max_length=300,
        description=(
            "The path of the field **inside the source** that this record "
            "was taken from, e.g. 'release_date.date' for a Steam release "
            "date, 'platforms.windows' for a platform statement, or "
            "'pc_requirements.minimum' for a PC requirement tier. It "
            "answers 'which field of which source does this record "
            "describe?', while ``field_path`` answers 'which draft field "
            "does this record support?'. The binding is **server-owned**: "
            "the service sets it when it mints the record, and a model may "
            "never author or change it. It is also **advisory**: it "
            "records where the fact was read from, so it is never a "
            "condition for verification, and a record without one is "
            "still a valid record."
        ),
    )
    retrieved_at: Optional[str] = Field(
        default=None,
        max_length=64,
        description="ISO-8601 timestamp of when the evidence was obtained.",
    )
    excerpt: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Verbatim excerpt backing the evidence entry.",
    )
    truncated: bool = Field(
        default=False,
        description=(
            "'True' when the source value was longer than the contract's "
            "field limit and was therefore shortened before it was recorded "
            "here. A truncated record describes only a **prefix** of the "
            "source fact, so it can never justify a 'verified' field: the "
            "value and the excerpt agree with each other, but neither is "
            "proof that the complete source fact was captured. The flag is "
            "**server-owned**: the service sets it when it mints the record, "
            "and a model can neither author nor clear it."
        ),
    )


class RequirementField(BaseModel):
    """One independently modelled system-requirement field.

    The field is self-describing: it always has a value slot (possibly
    ``None``) and always has an explicit verification status.
    """

    model_config = ConfigDict(extra="forbid")

    value: Optional[str] = Field(
        default=None,
        max_length=500,
        description=(
            "The requirement value, or null when unknown. Never a "
            "placeholder string such as 'unknown'."
        ),
    )
    verification: FieldVerification = Field(
        default_factory=FieldVerification,
        description="Field-level verification state.",
    )
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="Ids of the evidence entries backing this field.",
    )

    @field_validator("value", mode="before")
    @classmethod
    def _normalize_value(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("requirement value must be a string or null")
        return normalize_optional_text(value)

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def _normalize_evidence_ids(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("evidence_ids must be a list")
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        # Preserve order, drop duplicates.
        seen: set[str] = set()
        result: List[str] = []
        for item in cleaned:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    @model_validator(mode="after")
    def _align_status_with_value(self) -> "RequirementField":
        """Keep the status and the value consistent.

        * No value → the fact is unknown (a conflicting field is the one
          documented exception: it deliberately carries no winner).
        * A value → the status can no longer be "unknown".
        """

        if self.value is None:
            if self.verification.status in ("verified", "unverified"):
                self.verification.status = "unknown"
        else:
            if self.verification.status == "unknown":
                self.verification.status = "unverified"
        return self

    @property
    def is_known(self) -> bool:
        """``True`` when a value is present."""
        return self.value is not None


class RequirementTier(BaseModel):
    """One requirement tier (``minimum`` or ``recommended``).

    All six fields are modelled independently so a missing field never
    drags its siblings down with it.
    """

    model_config = ConfigDict(extra="forbid")

    os: RequirementField = Field(default_factory=RequirementField)
    processor: RequirementField = Field(default_factory=RequirementField)
    memory: RequirementField = Field(default_factory=RequirementField)
    graphics: RequirementField = Field(default_factory=RequirementField)
    storage: RequirementField = Field(default_factory=RequirementField)
    notes: RequirementField = Field(default_factory=RequirementField)

    def fields(self) -> Dict[str, RequirementField]:
        """Return the six fields keyed by name."""
        return {name: getattr(self, name) for name in REQUIREMENT_FIELD_NAMES}


class SystemRequirements(BaseModel):
    """Minimum and recommended system requirements.

    The two tiers are fully independent: a value in ``minimum`` is never
    propagated into ``recommended``.

    This model is deliberately *lenient* — it does not reject a
    recommended value that mirrors the minimum. The raw model output is
    an untrusted draft, and the service corrects that case
    deterministically instead of failing the whole call. The **public**
    response contract is the model that enforces the anti-inference rule.
    """

    model_config = ConfigDict(extra="forbid")

    minimum: RequirementTier = Field(default_factory=RequirementTier)
    recommended: RequirementTier = Field(default_factory=RequirementTier)


class ReleaseDate(BaseModel):
    """Release date. ISO-8601 date or ``None`` when unknown."""

    model_config = ConfigDict(extra="forbid")

    value: Optional[str] = Field(
        default=None,
        max_length=64,
        description="ISO-8601 date (YYYY-MM-DD), or null when unknown.",
    )
    verification: FieldVerification = Field(default_factory=FieldVerification)
    evidence_ids: List[str] = Field(default_factory=list)

    @field_validator("value", mode="before")
    @classmethod
    def _normalize_value(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("release_date must be a string or null")
        return normalize_iso_date(value)

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def _normalize_evidence_ids(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("evidence_ids must be a list")
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        seen: set[str] = set()
        result: List[str] = []
        for item in cleaned:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    @model_validator(mode="after")
    def _align_status_with_value(self) -> "ReleaseDate":
        if self.value is None:
            if self.verification.status in ("verified", "unverified"):
                self.verification.status = "unknown"
        else:
            if self.verification.status == "unknown":
                self.verification.status = "unverified"
        return self

    @property
    def is_known(self) -> bool:
        return self.value is not None


class Platforms(BaseModel):
    """Supported platforms. An empty list means "not known"."""

    model_config = ConfigDict(extra="forbid")

    values: List[str] = Field(
        default_factory=list,
        max_length=50,
        description="Platform names. An empty list means unknown.",
    )
    verification: FieldVerification = Field(default_factory=FieldVerification)
    evidence_ids: List[str] = Field(default_factory=list)

    @field_validator("values", mode="before")
    @classmethod
    def _normalize_values(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("platforms must be a list")
        cleaned: List[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                raise ValueError("platform entries must be strings")
            text = item.strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text)
        return cleaned

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def _normalize_evidence_ids(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("evidence_ids must be a list")
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        seen: set[str] = set()
        result: List[str] = []
        for item in cleaned:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    @model_validator(mode="after")
    def _align_status_with_value(self) -> "Platforms":
        if not self.values:
            if self.verification.status in ("verified", "unverified"):
                self.verification.status = "unknown"
        else:
            if self.verification.status == "unknown":
                self.verification.status = "unverified"
        return self

    @property
    def is_known(self) -> bool:
        return bool(self.values)


__all__ = [
    "ErrorResponse",
    "VerificationStatus",
    "EvidenceSourceType",
    "REQUIREMENT_FIELD_NAMES",
    "COMPLETENESS_REQUIREMENT_FIELDS",
    "UNKNOWN_SENTINELS",
    "normalize_optional_text",
    "normalize_iso_date",
    "is_utc_iso_timestamp",
    "FieldVerification",
    "EvidenceReference",
    "RequirementField",
    "RequirementTier",
    "SystemRequirements",
    "ReleaseDate",
    "Platforms",
]
