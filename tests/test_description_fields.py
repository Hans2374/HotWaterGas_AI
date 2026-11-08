"""Unit tests for the shared, version-neutral field module.

The module under test is :mod:`app.schemas.description_fields`. Two
things are worth asserting here, and they are different in kind:

1. **Behaviour.** The normalisation helpers and the value objects must
   behave exactly as they did before the extraction. The assertions
   below were written against the documented contract, not against the
   implementation, so they would catch a silent behaviour change.

2. **Schema identity.** The JSON Schema that is sent verbatim to the
   model provider names its definitions after the models' own class
   names. The shared models are version-neutral, so those names are
   neutral too, and the structure of the document is protected by a
   structural golden test rather than by a naming convention.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import pytest
from pydantic import BaseModel, ValidationError

from app.schemas.description_fields import (
    COMPLETENESS_REQUIREMENT_FIELDS,
    REQUIREMENT_FIELD_NAMES,
    UNKNOWN_SENTINELS,
    EvidenceReference,
    EvidenceSourceType,
    FieldVerification,
    Platforms,
    ReleaseDate,
    RequirementField,
    RequirementTier,
    SystemRequirements,
    VerificationStatus,
    is_utc_iso_timestamp,
    normalize_iso_date,
    normalize_optional_text,
)


#: A trailing version suffix on a generated definition name, e.g.
#: ``...V2`` or ``...V3``. The schema identities are version-neutral,
#: so a name that still carries a version suffix is a defect.
_VERSION_SUFFIX = re.compile(r"V\d+$")


# ── Normalisation helpers ─────────────────────────────────────────────────


class TestNormalizeOptionalText:
    def test_none_stays_none(self) -> None:
        assert normalize_optional_text(None) is None

    def test_a_non_string_becomes_none(self) -> None:
        assert normalize_optional_text(123) is None
        assert normalize_optional_text(["a"]) is None

    def test_surrounding_whitespace_is_removed(self) -> None:
        assert normalize_optional_text("  Elden Ring  ") == "Elden Ring"

    def test_an_empty_string_becomes_none(self) -> None:
        assert normalize_optional_text("") is None
        assert normalize_optional_text("   ") is None

    @pytest.mark.parametrize("sentinel", sorted(UNKNOWN_SENTINELS))
    def test_every_known_sentinel_becomes_none(self, sentinel: str) -> None:
        # The sentinel match is case-insensitive, so the upper-cased
        # spelling must be recognised too.
        assert normalize_optional_text(sentinel) is None
        assert normalize_optional_text(sentinel.upper()) is None

    def test_a_real_value_is_kept(self) -> None:
        assert normalize_optional_text("Windows 11") == "Windows 11"

    def test_the_sentinel_set_is_not_empty(self) -> None:
        # A guard against an accidental "empty set" regression, which
        # would make every string "unknown".
        assert len(UNKNOWN_SENTINELS) > 10
        assert "unknown" in UNKNOWN_SENTINELS
        assert "" in UNKNOWN_SENTINELS


class TestNormalizeIsoDate:
    def test_none_stays_none(self) -> None:
        assert normalize_iso_date(None) is None

    def test_an_iso_date_is_kept(self) -> None:
        assert normalize_iso_date("2026-01-31") == "2026-01-31"

    def test_surrounding_whitespace_is_removed(self) -> None:
        assert normalize_iso_date("  2026-01-31  ") == "2026-01-31"

    def test_a_timestamp_is_truncated_to_its_date(self) -> None:
        assert normalize_iso_date("2026-01-31T10:00:00Z") == "2026-01-31"
        assert normalize_iso_date("2026-01-31T10:00:00+00:00") == "2026-01-31"

    def test_an_unknown_sentinel_becomes_none(self) -> None:
        assert normalize_iso_date("unknown") is None
        assert normalize_iso_date("N/A") is None

    def test_unparseable_text_becomes_none(self) -> None:
        assert normalize_iso_date("not a date") is None
        assert normalize_iso_date("31/01/2026") is None

    def test_an_impossible_date_becomes_none(self) -> None:
        assert normalize_iso_date("2026-02-30") is None


class TestIsUtcIsoTimestamp:
    def test_a_utc_timestamp_is_accepted(self) -> None:
        assert is_utc_iso_timestamp("2026-10-27T08:15:00+00:00") is True
        assert is_utc_iso_timestamp("2026-10-27T08:15:00Z") is True

    def test_a_non_utc_offset_is_rejected(self) -> None:
        assert is_utc_iso_timestamp("2026-10-27T08:15:00+07:00") is False
        assert is_utc_iso_timestamp("2026-10-27T08:15:00-05:00") is False

    def test_a_naive_timestamp_is_rejected(self) -> None:
        # A naive local time names no instant, so it cannot say which
        # moment was verified.
        assert is_utc_iso_timestamp("2026-10-27T08:15:00") is False

    def test_a_date_without_a_time_is_rejected(self) -> None:
        assert is_utc_iso_timestamp("2026-10-27") is False

    def test_free_text_is_rejected(self) -> None:
        assert is_utc_iso_timestamp("not a timestamp") is False
        assert is_utc_iso_timestamp("") is False


# ── FieldVerification ─────────────────────────────────────────────────────


class TestFieldVerification:
    def test_the_status_defaults_to_unknown(self) -> None:
        assert FieldVerification().status == "unknown"

    def test_every_documented_status_is_accepted(self) -> None:
        for status in ("verified", "unverified", "conflicting", "unknown"):
            assert FieldVerification(status=status).status == status

    def test_an_unknown_status_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FieldVerification(status="probably")

    def test_a_checked_at_timestamp_must_be_utc(self) -> None:
        with pytest.raises(ValidationError):
            FieldVerification(checked_at="2026-10-27T08:15:00")

    def test_a_checked_at_timestamp_must_be_a_string(self) -> None:
        with pytest.raises(ValidationError):
            FieldVerification(checked_at=20261027)

    def test_extra_fields_are_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            FieldVerification(unexpected="x")

    def test_the_note_is_optional_and_bounded(self) -> None:
        assert FieldVerification().note is None
        with pytest.raises(ValidationError):
            FieldVerification(note="x" * 501)


# ── EvidenceReference ─────────────────────────────────────────────────────


class TestEvidenceReference:
    def test_the_source_type_defaults_to_unknown(self) -> None:
        assert EvidenceReference(evidence_id="e1").source_type == "unknown"

    def test_every_documented_source_type_is_accepted(self) -> None:
        for source_type in ("caller_supplied", "steam", "manual", "unknown"):
            record = EvidenceReference(evidence_id="e1", source_type=source_type)
            assert record.source_type == source_type

    def test_an_unknown_source_type_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvidenceReference(evidence_id="e1", source_type="guess")

    def test_the_evidence_id_is_required_and_bounded(self) -> None:
        with pytest.raises(ValidationError):
            EvidenceReference()
        with pytest.raises(ValidationError):
            EvidenceReference(evidence_id="x" * 201)

    def test_the_truncated_flag_defaults_to_false(self) -> None:
        assert EvidenceReference(evidence_id="e1").truncated is False

    def test_extra_fields_are_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            EvidenceReference(evidence_id="e1", verified=True)


# ── RequirementField ──────────────────────────────────────────────────────


class TestRequirementField:
    def test_an_empty_field_is_unknown(self) -> None:
        field = RequirementField()
        assert field.value is None
        assert field.verification.status == "unknown"
        assert field.is_known is False

    def test_a_value_without_a_status_becomes_unverified(self) -> None:
        field = RequirementField(value="8 GB")
        assert field.verification.status == "unverified"
        assert field.is_known is True

    def test_a_verified_status_is_downgraded_when_there_is_no_value(self) -> None:
        field = RequirementField(verification={"status": "verified"})
        assert field.verification.status == "unknown"

    def test_an_unknown_value_becomes_none(self) -> None:
        assert RequirementField(value="unknown").value is None

    def test_evidence_ids_are_deduplicated(self) -> None:
        field = RequirementField(value="8 GB", evidence_ids=["e1", "e1", " e2 "])
        assert field.evidence_ids == ["e1", "e2"]

    def test_evidence_ids_must_be_a_list(self) -> None:
        with pytest.raises(ValidationError):
            RequirementField(evidence_ids="e1")

    def test_a_non_string_value_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RequirementField(value=8)

    def test_the_value_is_bounded(self) -> None:
        with pytest.raises(ValidationError):
            RequirementField(value="x" * 501)


# ── RequirementTier / SystemRequirements ──────────────────────────────────


class TestRequirementTier:
    def test_every_field_is_independent(self) -> None:
        tier = RequirementTier()
        assert set(tier.fields()) == set(REQUIREMENT_FIELD_NAMES)
        for name in REQUIREMENT_FIELD_NAMES:
            assert getattr(tier, name).value is None

    def test_the_tier_does_not_infer_between_fields(self) -> None:
        tier = RequirementTier(os=RequirementField(value="Windows 11"))
        assert tier.os.value == "Windows 11"
        assert tier.processor.value is None
        assert tier.memory.value is None

    def test_extra_fields_are_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            RequirementTier(gpu="RTX 4090")


class TestSystemRequirements:
    def test_both_tiers_default_to_empty(self) -> None:
        requirements = SystemRequirements()
        assert requirements.minimum.os.value is None
        assert requirements.recommended.os.value is None

    def test_the_tiers_are_independent(self) -> None:
        requirements = SystemRequirements(
            minimum=RequirementTier(os=RequirementField(value="Windows 11"))
        )
        assert requirements.minimum.os.value == "Windows 11"
        assert requirements.recommended.os.value is None

    def test_a_mirrored_recommended_value_is_allowed(self) -> None:
        # The raw draft is lenient here; the *public* response contract
        # is the model that enforces the anti-inference rule.
        mirrored = RequirementTier(os=RequirementField(value="Windows 11"))
        requirements = SystemRequirements(
            minimum=RequirementTier(os=RequirementField(value="Windows 11")),
            recommended=mirrored,
        )
        assert requirements.recommended.os.value == "Windows 11"

    def test_extra_fields_are_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            SystemRequirements(medium=RequirementTier())


# ── ReleaseDate / Platforms ───────────────────────────────────────────────


class TestReleaseDate:
    def test_an_empty_release_date_is_unknown(self) -> None:
        release_date = ReleaseDate()
        assert release_date.value is None
        assert release_date.verification.status == "unknown"
        assert release_date.is_known is False

    def test_a_valid_date_is_normalised(self) -> None:
        release_date = ReleaseDate(value="2026-01-31T10:00:00Z")
        assert release_date.value == "2026-01-31"
        assert release_date.verification.status == "unverified"

    def test_an_unparseable_date_becomes_none(self) -> None:
        assert ReleaseDate(value="soon").value is None

    def test_a_non_string_date_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReleaseDate(value=20260131)

    def test_evidence_ids_are_deduplicated(self) -> None:
        assert ReleaseDate(evidence_ids=["e1", "e1"]).evidence_ids == ["e1"]


class TestPlatforms:
    def test_an_empty_platform_list_is_unknown(self) -> None:
        platforms = Platforms()
        assert platforms.values == []
        assert platforms.verification.status == "unknown"
        assert platforms.is_known is False

    def test_platforms_are_deduplicated_case_insensitively(self) -> None:
        platforms = Platforms(values=["Windows", "windows", " LINUX "])
        # The first spelling of a platform wins; the duplicates are
        # dropped, but the surviving spelling is the one that was
        # supplied first.
        assert platforms.values == ["Windows", "LINUX"]

    def test_a_platform_list_becomes_unverified(self) -> None:
        assert Platforms(values=["Windows"]).verification.status == "unverified"

    def test_a_verified_status_is_downgraded_for_an_empty_list(self) -> None:
        platforms = Platforms(verification={"status": "verified"})
        assert platforms.verification.status == "unknown"

    def test_non_string_entries_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Platforms(values=[1, 2])

    def test_the_list_is_bounded(self) -> None:
        with pytest.raises(ValidationError):
            Platforms(values=[f"p{index}" for index in range(51)])


# ── Schema identity ───────────────────────────────────────────────────────


class TestSchemaIdentity:
    """The generated schema names the models by their true names.

    The JSON Schema is sent verbatim to the model provider, so the
    names in it are observable. The shared models are version-neutral,
    and their Python class names are now exactly the names the
    document uses: the symbol a caller imports and the name the
    provider sees are the same name.

    Nothing needs to keep the two in sync any more, because there is
    only one name. What *does* need protecting is the **structure** of
    the document — the types, the bounds, the enumerations, the
    required sets and the reference graph — and that is protected by
    the structural golden test in
    :mod:`tests.test_description_pipeline`.
    """

    #: The models whose identity the document exposes.
    SHARED_MODELS = (
        FieldVerification,
        EvidenceReference,
        RequirementField,
        RequirementTier,
        SystemRequirements,
        ReleaseDate,
        Platforms,
    )

    @staticmethod
    def _parent_for(model: type[BaseModel]) -> type[BaseModel]:
        """Build a parent model that references ``model`` as a field.

        The annotation is supplied through ``__annotations__`` rather
        than written inline, because this module uses
        ``from __future__ import annotations`` and a local class
        annotation would then be an unresolvable *string*.
        """

        return type(
            "_Parent",
            (BaseModel,),
            {"__annotations__": {"field": model}},
        )

    @pytest.mark.parametrize("model", list(SHARED_MODELS))
    def test_a_model_keeps_its_own_name(self, model: type[BaseModel]) -> None:
        # The class name and the schema name are the same name.
        assert model.__name__ == model.__qualname__
        assert not _VERSION_SUFFIX.search(model.__name__), (
            f"stale version suffix in {model.__name__!r}"
        )

    @pytest.mark.parametrize("model", list(SHARED_MODELS))
    def test_a_nested_model_is_defined_under_its_own_name(
        self, model: type[BaseModel]
    ) -> None:
        schema = model.model_json_schema()
        # A model generated on its own is the *root* of the schema, so
        # its own name is the root title; the definitions in ``$defs``
        # are its nested models, and they carry their own names too.
        assert schema["title"] == model.__name__
        for key in schema.get("$defs", {}):
            assert not _VERSION_SUFFIX.search(key), (
                f"stale version suffix in {key!r}"
            )

    @pytest.mark.parametrize("model", list(SHARED_MODELS))
    def test_a_parent_model_refers_to_the_definition_by_its_own_name(
        self, model: type[BaseModel]
    ) -> None:
        # A parent model that references the shared model must produce
        # a ``$defs`` entry under that model's own name, and the
        # reference must point at it.
        schema = self._parent_for(model).model_json_schema()
        assert model.__name__ in schema["$defs"]
        assert schema["properties"]["field"]["$ref"].endswith(
            f"/$defs/{model.__name__}"
        )

    @pytest.mark.parametrize("model", list(SHARED_MODELS))
    def test_no_version_suffix_survives_in_a_generated_schema(
        self, model: type[BaseModel]
    ) -> None:
        schema = self._parent_for(model).model_json_schema()
        assert schema["title"] == "_Parent"
        for key in schema["$defs"]:
            assert not _VERSION_SUFFIX.search(key), (
                f"stale version suffix in {key!r}"
            )

    def test_the_neutral_symbols_are_the_ones_on_the_module(self) -> None:
        # The module-level bindings are the neutral names, and they are
        # the very same class objects.
        from app.schemas import description_fields

        assert description_fields.FieldVerification is FieldVerification
        assert description_fields.RequirementTier is RequirementTier
        assert description_fields.ReleaseDate is ReleaseDate
        assert description_fields.Platforms is Platforms

    def test_the_legacy_schema_module_is_gone(self) -> None:
        # The versioned schema modules have been removed. The neutral
        # module is the only home of the shared value objects.
        import importlib.util

        assert importlib.util.find_spec("app.schemas.description_v2") is None
        assert importlib.util.find_spec("app.schemas.description_v3") is None

    def test_a_rename_never_turns_into_a_new_type(self) -> None:
        # The class object is the class object: a name change must
        # never produce a different type.
        assert FieldVerification.__name__ == "FieldVerification"
        assert isinstance(FieldVerification(), FieldVerification)


# ── Shared constants ──────────────────────────────────────────────────────


class TestSharedConstants:
    def test_the_requirement_field_names(self) -> None:
        assert REQUIREMENT_FIELD_NAMES == (
            "os",
            "processor",
            "memory",
            "graphics",
            "storage",
            "notes",
        )

    def test_the_completeness_fields_are_a_strict_subset(self) -> None:
        assert set(COMPLETENESS_REQUIREMENT_FIELDS).issubset(
            set(REQUIREMENT_FIELD_NAMES)
        )
        assert "notes" not in COMPLETENESS_REQUIREMENT_FIELDS

    def test_the_vocabulary_types(self) -> None:
        # These are ``Literal`` aliases; the assertion documents the
        # exact vocabulary that the contract accepts.
        assert set(VerificationStatus.__args__) == {
            "verified",
            "unverified",
            "conflicting",
            "unknown",
        }
        assert set(EvidenceSourceType.__args__) == {
            "caller_supplied",
            "steam",
            "manual",
            "unknown",
        }
