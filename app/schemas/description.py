"""Request/response models for the description translation.

The models describe the translation contract. They are built on the
version-neutral value objects in
:mod:`app.schemas.description_fields`.

What the contract is
--------------------
The endpoint translates an **English** Steam store description into
Vietnamese. It is a *narrow* job: the model's only duty is to translate
the supplied source faithfully. It does not draft a description from
scattered facts, and it never invents one.

Three things shape the contract
--------------------------------
**A dedicated warning channel.** The contract has a way to say "the source
was truncated" or "the source was too short to reach the target". It
carries a ``warnings`` collection with **stable codes** and a
**severity**, so a consumer can react programmatically instead of
parsing prose. The warning collection is deliberately *not* overloaded
onto any other channel: a warning means "the input was incomplete".

The codes keep the **source** and the **output** apart.
``source_insufficient`` is a statement about the English source the
model was shown, and it is derived from the source blocks alone.
``translation_below_target`` is a statement about the translated text.
A short translation never implies a short source, and a short source
never implies the model misbehaved.

**An explicit description-source block.** The response carries *what was
translated*: which Steam fields contributed, how many blocks, whether the
source was truncated. A consumer can therefore tell a complete
translation from a partial one without re-deriving anything.

**An unverified-by-construction description.** A translated description
is prose. Translating it does not check it against the source, and the
contract says so: the description's verification status is always
``unverified`` and the contract *rejects* a payload that claims
otherwise. The deterministic facts (release date, platforms, Windows
requirements) keep their own verification states.

Anti-inference rules
--------------------
The contract enforces these rules:

* ``system_requirements.recommended`` is never derived from
  ``system_requirements.minimum``.
* A ``verified`` field references evidence that exists.
* A missing fact is ``null``, never the string ``"unknown"``.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import DEFAULT_MAX_PRODUCT_NAME_LENGTH
from app.schemas.description_fields import (
    EvidenceReference,
    FieldVerification,
    Platforms,
    ReleaseDate,
    RequirementField,
    RequirementTier,
    SystemRequirements,
    normalize_optional_text,
)

#: Version tag embedded in every payload.
SCHEMA_VERSION: str = "3"

#: How serious a warning is.
#:
#: ``"info"`` records something the consumer should know about but that
#: does not undermine the result. ``"warning"`` records something that
#: materially limits the result.
WarningSeverity = Literal["info", "warning"]

#: The stable warning codes the contract can emit.
#:
#: The codes are part of the contract: a consumer may branch on them, so
#: they never change meaning. A new condition gets a **new** code
#: rather than being folded into an existing one.
#:
#: ``source_truncated``
#:     The English source was longer than the source budget, so only the
#:     blocks that fitted were translated. The translation is a
#:     **partial** translation of the Steam description.
#: ``source_insufficient``
#:     The selected, deduplicated source carried less material than the
#:     target range needs. This is a statement about the **source**, and
#:     it is measured on the source blocks — never on the model's
#:     output. A short translation does not produce this warning.
#: ``translation_below_target``
#:     The translated text is shorter than the soft target range. This
#:     is a statement about the **output**. It is emitted whether or not
#:     the source was sufficient, so a consumer can tell "the model
#:     returned less than we asked for" from "there was not enough to
#:     translate".
#: ``description_over_ceiling``
#:     The translated description exceeded the hard ceiling and the call
#:     failed. The draft is never truncated to fit.
#: ``source_block_dropped``
#:     At least one source block did not fit inside the source budget.
#:
#: There is deliberately **no** ``source_missing`` code. The
#: no-description path is a controlled HTTP error (a 502 with the
#: ``steam_unavailable`` code), not a successful response, so a warning
#: code for it could never appear in a delivered body. A code a
#: consumer can never observe is a code a consumer cannot rely on.
WARNING_CODES: tuple[str, ...] = (
    "source_truncated",
    "source_insufficient",
    "translation_below_target",
    "description_over_ceiling",
    "source_block_dropped",
)


class DescriptionWarning(BaseModel):
    """One admin-facing warning.

    The warning carries a **stable code** and a **severity**, plus a
    short human-readable message. The message is informational: a
    consumer should branch on ``code``, never on the text.
    """

    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description=(
            "A stable machine-readable code. See WARNING_CODES for the "
            "documented vocabulary."
        ),
    )
    severity: WarningSeverity = Field(
        default="warning",
        description="'info' or 'warning'.",
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description=(
            "A short, human-readable explanation for the admin. Never a "
            "fact about the product."
        ),
    )
    field_path: Optional[str] = Field(
        default=None,
        max_length=300,
        description=(
            "The dotted field path the warning concerns, when it concerns "
            "one."
        ),
    )


class DescriptionSourceBlock(BaseModel):
    """What the description source contributed.

    The block is **service-owned**: it describes the input the model
    was shown, not anything the model produced. A model can neither
    author nor widen it.
    """

    model_config = ConfigDict(extra="forbid")

    primary_field: Optional[str] = Field(
        default=None,
        max_length=64,
        description=(
            "The Steam field the primary blocks came from "
            "('detailed_description', or 'about_the_game' when the "
            "detailed description was absent). Null when no source "
            "carried any content."
        ),
    )
    supplementary_fields: List[str] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "The Steam fields that contributed additional blocks, e.g. "
            "'about_the_game' or 'short_description'."
        ),
    )
    skipped_fields: List[str] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "The Steam fields that were consulted but contributed "
            "nothing, because their content was already present."
        ),
    )
    block_count: int = Field(
        default=0,
        ge=0,
        description="The number of blocks that were translated.",
    )
    source_bytes: int = Field(
        default=0,
        ge=0,
        description="The size of the translated source, in UTF-8 bytes.",
    )
    total_source_bytes: int = Field(
        default=0,
        ge=0,
        description=(
            "The size of every parsed block, before the source budget was "
            "applied. A value larger than 'source_bytes' means blocks were "
            "dropped."
        ),
    )
    truncated: bool = Field(
        default=False,
        description=(
            "True when the source budget dropped at least one block. The "
            "translation is then a *partial* translation of the Steam "
            "description, and the response says so."
        ),
    )
    dropped_blocks: int = Field(
        default=0,
        ge=0,
        description="The number of blocks the source budget dropped.",
    )
    source_token_count: int = Field(
        default=0,
        ge=0,
        description=(
            "The number of whitespace-separated tokens in the English "
            "source that was shown to the model. This is a **source** "
            "measurement: it says nothing about how long the "
            "translation is."
        ),
    )
    source_sufficient: bool = Field(
        default=True,
        description=(
            "Whether the selected source carried enough material to "
            "reasonably support the target length. The assessment is "
            "deterministic and is made on the source blocks, never on "
            "the model's output: a short translation does not make the "
            "source insufficient. A truncated source is never sufficient. "
            "The assessment only selects a warning; it never rejects a "
            "faithful short translation."
        ),
    )
    min_source_tokens: int = Field(
        default=0,
        ge=0,
        description=(
            "The threshold the source token count was compared against. "
            "Zero means the assessment was not performed."
        ),
    )


class TranslatedDescription(BaseModel):
    """The translated Vietnamese description text and its metadata.

    The verification status is **always** ``unverified``. A translation
    that completed is not a fact that was checked: the contract enforces
    this, so a payload that claims ``verified`` is rejected outright
    rather than being quietly downgraded.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        description="The Vietnamese translation of the English source.",
    )
    language: str = Field(default="vi", max_length=16)
    word_count: int = Field(
        default=0,
        ge=0,
        description=(
            "The number of whitespace-separated tokens in the text. "
            "Server-owned: the model may not author it, and the service "
            "recomputes it from the text it ships."
        ),
    )
    within_target_range: bool = Field(
        default=False,
        description=(
            "True when the token count sits inside the soft target range. "
            "Server-owned, and purely informational."
        ),
    )
    verification: FieldVerification = Field(default_factory=FieldVerification)
    evidence_ids: List[str] = Field(default_factory=list)

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def _normalize_evidence_ids(cls, value: object) -> List[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("evidence_ids must be a list")
        cleaned: List[str] = []
        seen: set[str] = set()
        for item in value:
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            cleaned.append(text)
        return cleaned

    @model_validator(mode="after")
    def _force_unverified(self) -> "TranslatedDescription":
        """A translated description is never ``verified``.

        The service translates; it does not check the translation against
        the source. Claiming otherwise would read as a factual guarantee
        the pipeline cannot make, so the claim is refused at the contract
        boundary rather than being silently downgraded.
        """

        if self.verification.status == "verified":
            raise ValueError(
                "a translated description cannot be 'verified': the "
                "translation was not checked against the source"
            )
        if self.verification.status == "unknown":
            # The text always carries content, so the fact is "present
            # but unverified" rather than "unknown".
            self.verification.status = "unverified"
        return self


class DescriptionTranslationDraft(BaseModel):
    """The payload the model is asked to produce.

    It contains the translated text and nothing else. Every other
    field of the response - the deterministic Steam facts, the
    description-source block, the warnings - is owned by the service.
    """

    model_config = ConfigDict(extra="ignore")

    description: TranslatedDescription


class ModelMetadata(BaseModel):
    """Which model produced the translation, and how it behaved."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    provider: str = Field(..., min_length=1, max_length=64)
    model_id: str = Field(..., min_length=1, max_length=200)
    prompt_version: str = Field(..., min_length=1, max_length=64)
    duration_ms: Optional[int] = Field(default=None, ge=0)
    finish_reason: Optional[str] = Field(default=None, max_length=64)
    generated_at: Optional[str] = Field(default=None, max_length=64)


class GenerateDescriptionRequest(BaseModel):
    """Payload accepted by the translation seam.

    The request is deliberately minimal: a product name (used only to
    check the retrieved Steam title against the admin's intent) and an
    optional Steam App ID. The English description itself is retrieved
    from Steam; the caller never supplies it, so a caller cannot smuggle
    an instruction into the source that is about to be translated.
    """

    model_config = ConfigDict(extra="forbid")

    product_name: str = Field(
        ...,
        min_length=1,
        max_length=DEFAULT_MAX_PRODUCT_NAME_LENGTH,
        description="The product's official name (required).",
    )
    steam_app_id: Optional[int] = Field(
        default=None,
        ge=1,
        le=2_147_483_647,
        description=(
            "Optional Steam App ID. When present the service retrieves the "
            "English store metadata for this app and translates its "
            "description. The App ID is asserted by the caller and is "
            "validated against the retrieved title; it is never inferred "
            "from the product name."
        ),
    )

    @field_validator("steam_app_id", mode="before")
    @classmethod
    def _check_steam_app_id(cls, value: object) -> Optional[int]:
        """Validate the Steam App ID strictly.

        Only a genuine integer (or a string holding one) is accepted.
        ``bool`` is rejected explicitly: it is an ``int`` subclass in
        Python but ``True`` is not a valid App ID.
        """

        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("steam_app_id must be an integer")
        if isinstance(value, str):
            text = value.strip()
            if not text.isdigit():
                raise ValueError("steam_app_id must be an integer")
            value = int(text)
        if isinstance(value, float):
            raise ValueError("steam_app_id must be an integer")
        if not isinstance(value, int):
            raise ValueError("steam_app_id must be an integer")
        if value < 1:
            raise ValueError("steam_app_id must be a positive integer")
        if value > 2_147_483_647:
            raise ValueError("steam_app_id is out of range")
        return value

    @field_validator("product_name")
    @classmethod
    def _check_product_name(cls, value: str) -> str:
        cleaned = normalize_optional_text(value)
        if cleaned is None:
            raise ValueError("product_name must not be empty or whitespace")
        return cleaned


class GenerateDescriptionResponse(BaseModel):
    """The response contract.

    It carries the translated description, the deterministic Steam facts,
    the description-source metadata and the warning collection.

    The contract rules it enforces:

    * a ``verified`` field references evidence that exists,
    * the translated description is never ``verified``,
    * a recommended requirement never mirrors the minimum unless it
      references its own evidence, and
    * every evidence id is unique.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    schema_version: str = Field(default=SCHEMA_VERSION, max_length=16)
    description: TranslatedDescription
    release_date: ReleaseDate = Field(default_factory=ReleaseDate)
    platforms: Platforms = Field(default_factory=Platforms)
    system_requirements: SystemRequirements = Field(
        default_factory=SystemRequirements
    )
    description_source: DescriptionSourceBlock = Field(
        default_factory=DescriptionSourceBlock
    )
    warnings: List[DescriptionWarning] = Field(
        default_factory=list,
        max_length=50,
        description=(
            "Admin-facing warnings about the *input*, not about the "
            "product's facts. Every warning carries a stable code and a "
            "severity."
        ),
    )
    evidence: List[EvidenceReference] = Field(default_factory=list)
    model: ModelMetadata
    retrieved_at: Optional[str] = Field(
        default=None,
        max_length=64,
        description="ISO-8601 timestamp of the Steam retrieval.",
    )

    # ── Contract rules ─────────────────────────────────────────────────

    def _known_evidence_ids(self) -> set[str]:
        return {item.evidence_id for item in self.evidence}

    def _tracked_fields(self) -> list[tuple[str, object]]:
        tracked: list[tuple[str, object]] = [
            ("description", self.description),
            ("release_date", self.release_date),
            ("platforms", self.platforms),
        ]
        for tier_name in ("minimum", "recommended"):
            tier = getattr(self.system_requirements, tier_name)
            for field_name in tier.fields():
                tracked.append(
                    (f"system_requirements.{tier_name}.{field_name}",
                     getattr(tier, field_name))
                )
        return tracked

    @model_validator(mode="after")
    def _require_resolvable_field_evidence(self) -> "GenerateDescriptionResponse":
        """Every field evidence reference must resolve to a real record."""

        known_ids = self._known_evidence_ids()
        for path, field in self._tracked_fields():
            missing = [
                evidence_id
                for evidence_id in (field.evidence_ids or [])
                if evidence_id not in known_ids
            ]
            if missing:
                raise ValueError(
                    f"{path} references evidence that is not present in the "
                    f"evidence index: {', '.join(sorted(missing))}"
                )
        return self

    @model_validator(mode="after")
    def _require_evidence_for_verified_fields(self) -> "GenerateDescriptionResponse":
        """A ``verified`` field must reference evidence that exists."""

        known_ids = self._known_evidence_ids()
        for path, field in self._tracked_fields():
            if field.verification.status != "verified":
                continue
            if not field.evidence_ids:
                raise ValueError(
                    f"{path} is marked verified but references no evidence"
                )
            missing = [
                evidence_id
                for evidence_id in field.evidence_ids
                if evidence_id not in known_ids
            ]
            if missing:
                raise ValueError(
                    f"{path} references evidence that is not present in the "
                    "evidence index"
                )
        return self

    @model_validator(mode="after")
    def _require_unique_evidence_ids(self) -> "GenerateDescriptionResponse":
        """An evidence id must identify exactly one record."""

        seen: set[str] = set()
        for item in self.evidence:
            if item.evidence_id in seen:
                raise ValueError(
                    f"evidence id {item.evidence_id!r} appears more than "
                    "once; evidence ids must be unique"
                )
            seen.add(item.evidence_id)
        return self

    @model_validator(mode="after")
    def _forbid_copying_minimum_into_recommended(self) -> "GenerateDescriptionResponse":
        """Reject a recommended value that merely mirrors the minimum."""

        minimum_fields = self.system_requirements.minimum.fields()
        recommended_fields = self.system_requirements.recommended.fields()
        for name, minimum_field in minimum_fields.items():
            recommended_field = recommended_fields[name]
            if minimum_field.value is None or recommended_field.value is None:
                continue
            if minimum_field.value.strip().casefold() != recommended_field.value.strip().casefold():
                continue
            if recommended_field.evidence_ids:
                continue
            raise ValueError(
                f"system_requirements.recommended.{name} must not mirror "
                f"system_requirements.minimum.{name} without its own evidence"
            )
        return self

    @model_validator(mode="after")
    def _require_known_warning_codes(self) -> "GenerateDescriptionResponse":
        """Every warning must carry a documented code.

        An undocumented code would be a code no consumer can branch on,
        which makes the warning collection useless for its only purpose.
        """

        for warning in self.warnings:
            if warning.code not in WARNING_CODES:
                raise ValueError(
                    f"unknown warning code {warning.code!r}; documented "
                    f"codes are: {', '.join(WARNING_CODES)}"
                )
        return self


__all__ = [
    "SCHEMA_VERSION",
    "WARNING_CODES",
    "WarningSeverity",
    "DescriptionWarning",
    "DescriptionSourceBlock",
    "TranslatedDescription",
    "DescriptionTranslationDraft",
    "ModelMetadata",
    "GenerateDescriptionRequest",
    "GenerateDescriptionResponse",
]
