"""Tests: deterministic Steam fact extraction.

Four facts are extracted out of the model and into the parser:
the release date, the platform availability and the two PC
requirement tiers. These tests pin the behaviour that makes that
extraction trustworthy.

The rules the tests enforce:

* a date is parsed only when it is **unambiguous** — every supported
  locale spelling is accepted, and every ambiguous one stays unknown,
* platform availability keeps **three** states apart: supported,
  explicitly unsupported, and not reported,
* the two requirement tiers are populated only from their own source
  key, and a missing recommended tier is never derived from the
  minimum tier,
* the macOS and Linux requirement tiers are never mapped, and
* every extracted fact traces back to the Steam field it came from.

Everything here is a pure function call against an inline payload:
no network, no clock, no model.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from app.retrieval.steam_mapper import map_appdetails, parse_steam_date


APP_ID = 1245620
RETRIEVED_AT = "2026-01-01T00:00:00+00:00"


def _map(payload: Dict[str, Any], app_id: int = APP_ID) -> Any:
    """Map an inline payload into ``(facts, evidence, evidence_by_field)``."""

    return map_appdetails(app_id, payload, retrieved_at=RETRIEVED_AT)


def _payload(**data: Any) -> Dict[str, Any]:
    """Wrap ``data`` in Steam's ``appdetails`` envelope."""

    return {str(APP_ID): {"success": True, "data": data}}


# ── B1. Release dates ─────────────────────────────────────────────────────


class TestVietnameseReleaseDates:
    """Steam's Vietnamese locale abbreviates the month and attaches the
    month number to it."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("13 Thg11, 2007", "2007-11-13"),
            ("13 Thg 11, 2007", "2007-11-13"),
            ("13 thg11, 2007", "2007-11-13"),
            ("13 THG11, 2007", "2007-11-13"),
            ("13 Thg 11,2007", "2007-11-13"),
            ("13 Thg3, 2022", "2022-03-13"),
            ("13 Thg 3, 2022", "2022-03-13"),
            ("25 Thg 3, 2022", "2022-03-25"),
            ("25 thg3, 2022", "2022-03-25"),
            ("13 Thang 11, 2007", "2007-11-13"),
            ("13 thang11, 2007", "2007-11-13"),
        ],
    )
    def test_vietnamese_dates_become_iso(self, raw: str, expected: str) -> None:
        assert parse_steam_date(raw) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Nov 13, 2007", "2007-11-13"),
            ("Mar 25, 2022", "2022-03-25"),
            ("Feb 25, 2022", "2022-02-25"),
            ("13 Nov, 2007", "2007-11-13"),
            ("December 1, 1999", "1999-12-01"),
            ("Sept 9, 2007", "2007-09-09"),
            ("2022-02-25", "2022-02-25"),
        ],
    )
    def test_the_english_forms_still_work(self, raw: str, expected: str) -> None:
        assert parse_steam_date(raw) == expected


class TestAmbiguousDatesStayUnknown:
    """A date that cannot be parsed with confidence stays unknown.

    A wrong date is worse than a missing one, so anything ambiguous is
    refused rather than guessed at.
    """

    @pytest.mark.parametrize(
        "raw",
        [
            # Two-digit years are ambiguous about the century.
            "13 Thg11, 07",
            "13 Thg 11, 07",
            # A day that does not exist in that month.
            "31 Thg 2, 2022",
            "31 Thg11, 2007",
            # A day or a month outside its range.
            "32 Thg 11, 2007",
            "0 Thg 11, 2007",
            "13 Thg 13, 2007",
            "13 Thg 0, 2007",
            # An unknown month.
            "13 Blah 11, 2007",
            "13 Xyz, 2007",
            # A missing part.
            "13 Thg11",
            "13 Thg 11",
            "Thg 11, 2007",
            "2007",
            # A year that is not plausible for a release.
            "13 Thg11, 999",
            "13 Thg11, 12007",
            # The unreleased markers.
            "Coming soon",
            "To be announced",
            "TBA",
            "",
            "not a date",
        ],
    )
    def test_unparsable_dates_stay_none(self, raw: str) -> None:
        assert parse_steam_date(raw) is None

    @pytest.mark.parametrize("raw", [None, 123, 13.5, True, {}, [], object()])
    def test_non_string_input_stays_none(self, raw: Any) -> None:
        assert parse_steam_date(raw) is None

    def test_a_two_digit_year_is_not_resolved_by_guessing(self) -> None:
        # "13 Thg11, 07" could be 2007 or 1907. It stays unknown.
        assert parse_steam_date("13 Thg11, 07") is None


class TestReleaseDateInTheFacts:
    """The parsed date reaches the facts and its evidence."""

    def test_a_vietnamese_date_is_mapped(self) -> None:
        facts, _, _ = _map(
            _payload(name="X", release_date={"coming_soon": False, "date": "13 Thg11, 2007"})
        )
        assert facts.release_date == "2007-11-13"

    def test_a_coming_soon_date_stays_unknown(self) -> None:
        facts, _, _ = _map(
            _payload(name="X", release_date={"coming_soon": True, "date": "Coming soon"})
        )
        assert facts.release_date is None

    def test_an_unparsable_date_stays_unknown(self) -> None:
        facts, _, _ = _map(
            _payload(name="X", release_date={"date": "sometime in 2007"})
        )
        assert facts.release_date is None

    def test_a_missing_date_stays_unknown(self) -> None:
        facts, _, _ = _map(_payload(name="X"))
        assert facts.release_date is None


# ── B2. Platforms ─────────────────────────────────────────────────────────


class TestPlatformAvailability:
    """Platform availability keeps three states apart."""

    def test_supported_platforms_are_listed(self) -> None:
        facts, _, _ = _map(
            _payload(platforms={"windows": True, "mac": True, "linux": False})
        )
        assert facts.platforms == ["Windows", "macOS"]

    def test_an_unsupported_platform_is_not_listed(self) -> None:
        facts, _, _ = _map(
            _payload(platforms={"windows": True, "mac": False, "linux": False})
        )
        assert facts.platforms == ["Windows"]

    def test_unsupported_and_unreported_stay_distinct(self) -> None:
        # Linux was explicitly reported as unsupported; macOS was not
        # reported at all. They are different facts.
        facts, _, _ = _map(
            _payload(platforms={"windows": True, "mac": None, "linux": False})
        )
        availability = facts.platform_availability
        assert availability.windows is True
        assert availability.mac is None
        assert availability.linux is False

    def test_all_platform_keys_missing_is_all_unknown(self) -> None:
        facts, _, _ = _map(_payload(platforms={}))
        availability = facts.platform_availability
        assert facts.platforms == []
        assert availability.windows is None
        assert availability.mac is None
        assert availability.linux is None

    def test_a_null_value_is_not_reported(self) -> None:
        facts, _, _ = _map(
            _payload(platforms={"windows": None, "mac": None, "linux": None})
        )
        assert facts.platforms == []
        assert facts.platform_availability.windows is None

    def test_a_missing_platforms_object_is_all_unknown(self) -> None:
        facts, _, _ = _map(_payload(name="X"))
        assert facts.platforms == []
        assert facts.platform_availability.windows is None
        assert facts.platform_availability.mac is None
        assert facts.platform_availability.linux is None

    @pytest.mark.parametrize(
        "raw", ["Windows", 123, None, [], ["windows"], True]
    )
    def test_an_unexpected_platforms_type_is_all_unknown(self, raw: Any) -> None:
        facts, _, _ = _map(_payload(platforms=raw))
        assert facts.platforms == []
        assert facts.platform_availability.windows is None

    @pytest.mark.parametrize(
        "raw", ["true", "yes", 1, "1", ["windows"], {"nested": True}]
    )
    def test_a_non_boolean_value_is_not_reported(self, raw: Any) -> None:
        # Steam's contract for this object is a boolean. A truthy
        # non-boolean is not Steam saying "yes".
        facts, _, _ = _map(_payload(platforms={"windows": raw, "mac": False}))
        assert facts.platforms == []
        assert facts.platform_availability.windows is None
        assert facts.platform_availability.mac is False

    def test_platform_support_is_never_inferred_from_requirements(self) -> None:
        # The macOS tier exists, but that does not make the game a
        # macOS product.
        facts, _, _ = _map(
            _payload(
                platforms={"windows": False, "mac": False, "linux": False},
                mac_requirements={"minimum": {"os": "macOS 12"}},
                linux_requirements={"minimum": {"os": "Ubuntu 22.04"}},
            )
        )
        assert facts.platforms == []
        assert facts.platform_availability.mac is False

    def test_the_availability_defaults_to_unknown(self) -> None:
        facts, _, _ = _map(_payload(name="X"))
        availability = facts.platform_availability
        assert availability.windows is None
        assert availability.mac is None
        assert availability.linux is None


# ── B3. PC requirements ───────────────────────────────────────────────────


class TestPcRequirementExtraction:
    """Only the PC/Windows tiers are mapped."""

    def test_both_tiers_are_mapped(self) -> None:
        facts, _, _ = _map(
            _payload(
                pc_requirements={
                    "minimum": {
                        "os": "Windows 10 64-bit",
                        "processor": "Intel Core i5",
                        "memory": "8 GB RAM",
                        "graphics": "GTX 960",
                        "storage": "60 GB",
                    },
                    "recommended": {
                        "os": "Windows 10 64-bit",
                        "processor": "Intel Core i7",
                        "memory": "16 GB RAM",
                        "graphics": "GTX 1080",
                        "storage": "60 GB",
                    },
                }
            )
        )
        assert facts.minimum.os == "Windows 10 64-bit"
        assert facts.minimum.processor == "Intel Core i5"
        assert facts.minimum.memory == "8 GB RAM"
        assert facts.minimum.graphics == "GTX 960"
        assert facts.minimum.storage == "60 GB"
        assert facts.recommended.processor == "Intel Core i7"
        assert facts.recommended.memory == "16 GB RAM"

    def test_minimum_only_leaves_recommended_unknown(self) -> None:
        facts, _, _ = _map(
            _payload(
                pc_requirements={
                    "minimum": {"os": "Windows 10", "memory": "8 GB RAM"}
                }
            )
        )
        assert facts.minimum.os == "Windows 10"
        assert facts.minimum.memory == "8 GB RAM"
        # Nothing was invented for the recommended tier.
        assert facts.recommended.os is None
        assert facts.recommended.processor is None
        assert facts.recommended.memory is None
        assert facts.recommended.graphics is None
        assert facts.recommended.storage is None

    def test_recommended_is_never_copied_from_minimum(self) -> None:
        facts, _, _ = _map(
            _payload(
                pc_requirements={
                    "minimum": {
                        "os": "Windows 10",
                        "processor": "Intel Core i5",
                        "memory": "8 GB RAM",
                    }
                }
            )
        )
        assert facts.minimum.memory == "8 GB RAM"
        assert facts.recommended.memory != facts.minimum.memory
        assert facts.recommended.memory is None

    def test_both_tiers_absent(self) -> None:
        facts, _, _ = _map(_payload(pc_requirements={}))
        assert facts.minimum.os is None
        assert facts.recommended.os is None

    def test_the_whole_pc_requirements_object_absent(self) -> None:
        facts, _, _ = _map(_payload(name="X"))
        assert facts.minimum.os is None
        assert facts.recommended.os is None

    @pytest.mark.parametrize("empty", ["", "   ", {}, None])
    def test_empty_and_null_tiers_produce_nothing(self, empty: Any) -> None:
        facts, _, _ = _map(_payload(pc_requirements={"minimum": empty}))
        assert facts.minimum.os is None
        assert facts.minimum.processor is None
        assert facts.recommended.os is None

    def test_missing_individual_fields_stay_missing(self) -> None:
        facts, _, _ = _map(
            _payload(pc_requirements={"minimum": {"os": "Windows 10"}})
        )
        assert facts.minimum.os == "Windows 10"
        assert facts.minimum.processor is None
        assert facts.minimum.memory is None
        assert facts.minimum.graphics is None
        assert facts.minimum.storage is None

    def test_an_empty_string_field_is_treated_as_missing(self) -> None:
        facts, _, _ = _map(
            _payload(
                pc_requirements={"minimum": {"os": "", "memory": "8 GB RAM"}}
            )
        )
        assert facts.minimum.os is None
        assert facts.minimum.memory == "8 GB RAM"

    def test_html_content_is_cleaned(self) -> None:
        facts, _, _ = _map(
            _payload(
                pc_requirements={
                    "minimum": {
                        "os": "Windows 10 64-bit<br><strong>(64-bit)</strong>",
                        "processor": "Requires&nbsp;Intel&nbsp;Core i5",
                        "memory": "8&nbsp;GB RAM",
                    }
                }
            )
        )
        assert facts.minimum.os == "Windows 10 64-bit (64-bit)"
        assert facts.minimum.processor == "Requires Intel Core i5"
        assert facts.minimum.memory == "8 GB RAM"


class TestSteamStringRequirementFormat:
    """Steam sometimes serves a tier as one HTML string."""

    def test_the_string_format_is_parsed(self) -> None:
        facts, _, _ = _map(
            _payload(
                pc_requirements={
                    "minimum": (
                        "<strong>Operating System:</strong> Windows 10 64-bit<br>"
                        "<strong>Processor:</strong> Intel Core i5<br>"
                        "<strong>Memory:</strong> 8 GB RAM<br>"
                        "<strong>Graphics:</strong> GTX 960<br>"
                        "<strong>Storage:</strong> 60 GB"
                    )
                }
            )
        )
        assert facts.minimum.os == "Windows 10 64-bit"
        assert facts.minimum.processor == "Intel Core i5"
        assert facts.minimum.memory == "8 GB RAM"
        assert facts.minimum.graphics == "GTX 960"
        assert facts.minimum.storage == "60 GB"

    def test_the_string_format_is_parsed_with_newlines(self) -> None:
        facts, _, _ = _map(
            _payload(
                pc_requirements={
                    "minimum": "Operating System: Windows 10\nMemory: 8 GB RAM"
                }
            )
        )
        assert facts.minimum.os == "Windows 10"
        assert facts.minimum.memory == "8 GB RAM"

    def test_an_unknown_label_is_ignored_rather_than_guessed(self) -> None:
        facts, _, _ = _map(
            _payload(
                pc_requirements={
                    "minimum": (
                        "Operating System: Windows 10<br>"
                        "Additional Notes: A note that is not a requirement"
                    )
                }
            )
        )
        assert facts.minimum.os == "Windows 10"
        # The note is not a modelled field, so it is dropped.
        assert facts.minimum.memory is None
        assert facts.minimum.processor is None

    def test_a_line_without_a_colon_is_ignored(self) -> None:
        facts, _, _ = _map(
            _payload(
                pc_requirements={
                    "minimum": "Operating System: Windows 10<br>no colon here"
                }
            )
        )
        assert facts.minimum.os == "Windows 10"
        assert facts.minimum.memory is None

    def test_a_label_with_no_value_is_ignored(self) -> None:
        facts, _, _ = _map(
            _payload(
                pc_requirements={"minimum": "Operating System: Windows 10<br>Memory:"}
            )
        )
        assert facts.minimum.os == "Windows 10"
        assert facts.minimum.memory is None

    def test_the_string_format_never_fills_the_recommended_tier(self) -> None:
        facts, _, _ = _map(
            _payload(
                pc_requirements={
                    "minimum": "Operating System: Windows 10<br>Memory: 8 GB RAM"
                }
            )
        )
        assert facts.recommended.os is None
        assert facts.recommended.memory is None


class TestMacAndLinuxRequirementsAreNeverMapped:
    """The system-requirements section is PC/Windows only."""

    def test_the_macos_tier_never_enters_the_result(self) -> None:
        facts, evidence, _ = _map(
            _payload(
                pc_requirements={"minimum": {"os": "Windows 10"}},
                mac_requirements={
                    "minimum": {
                        "os": "macOS 12",
                        "processor": "Apple M1",
                        "memory": "16 GB RAM",
                    }
                },
            )
        )
        assert facts.minimum.os == "Windows 10"
        assert facts.minimum.processor is None
        assert facts.minimum.memory is None
        rendered = json.dumps(
            {
                "facts": facts.__dict__,
                "evidence": [item.model_dump() for item in evidence],
            },
            default=str,
        )
        assert "macOS" not in rendered
        assert "Apple M1" not in rendered

    def test_the_linux_tier_never_enters_the_result(self) -> None:
        facts, evidence, _ = _map(
            _payload(
                pc_requirements={"minimum": {"os": "Windows 10"}},
                linux_requirements={
                    "minimum": {"os": "Ubuntu 22.04", "processor": "Intel Core i5"}
                },
            )
        )
        assert facts.minimum.os == "Windows 10"
        rendered = json.dumps(
            {
                "facts": facts.__dict__,
                "evidence": [item.model_dump() for item in evidence],
            },
            default=str,
        )
        assert "Ubuntu" not in rendered

    def test_the_macos_tier_alone_does_not_populate_the_pc_tier(self) -> None:
        # No pc_requirements at all: the macOS tier must not stand in
        # for it.
        facts, _, _ = _map(
            _payload(mac_requirements={"minimum": {"os": "macOS 12"}})
        )
        assert facts.minimum.os is None
        assert facts.recommended.os is None

    def test_no_evidence_names_a_macos_or_linux_source(self) -> None:
        _, evidence, _ = _map(
            _payload(
                pc_requirements={"minimum": {"os": "Windows 10"}},
                mac_requirements={"minimum": {"os": "macOS 12"}},
                linux_requirements={"minimum": {"os": "Ubuntu 22.04"}},
            )
        )
        for item in evidence:
            assert item.source_field is None or "mac" not in item.source_field
            assert item.source_field is None or "linux" not in item.source_field


# ── B4. Provenance and evidence ───────────────────────────────────────────


class TestEvidenceProvenance:
    """Every extracted fact names the Steam field it came from."""

    def test_the_release_date_names_its_source_field(self) -> None:
        _, evidence, evidence_by_field = _map(
            _payload(name="X", release_date={"date": "13 Thg11, 2007"})
        )
        record = _record_for(evidence, evidence_by_field, "release_date")
        assert record.source_field == "release_date.date"

    def test_the_platform_list_names_its_source_field(self) -> None:
        _, evidence, evidence_by_field = _map(
            _payload(platforms={"windows": True, "mac": False, "linux": False})
        )
        record = _record_for(evidence, evidence_by_field, "platforms")
        # The draft models the platform list as one field, so the
        # record is bound to "platforms" — but it names the individual
        # Steam keys the list was built from, so each platform
        # statement is traceable to its own source key.
        assert record.source_field == "platforms.windows"

    def test_the_platform_list_names_every_supported_source_key(self) -> None:
        _, evidence, evidence_by_field = _map(
            _payload(platforms={"windows": True, "mac": True, "linux": False})
        )
        record = _record_for(evidence, evidence_by_field, "platforms")
        assert record.source_field == "platforms.windows, platforms.mac"

    def test_the_minimum_tier_names_its_source_field(self) -> None:
        _, evidence, evidence_by_field = _map(
            _payload(pc_requirements={"minimum": {"os": "Windows 10"}})
        )
        record = _record_for(
            evidence, evidence_by_field, "system_requirements.minimum.os"
        )
        assert record.source_field == "pc_requirements.minimum"

    def test_the_recommended_tier_names_its_source_field(self) -> None:
        _, evidence, evidence_by_field = _map(
            _payload(pc_requirements={"recommended": {"os": "Windows 10"}})
        )
        record = _record_for(
            evidence, evidence_by_field, "system_requirements.recommended.os"
        )
        assert record.source_field == "pc_requirements.recommended"

    def test_the_two_tiers_name_different_source_fields(self) -> None:
        _, evidence, evidence_by_field = _map(
            _payload(
                pc_requirements={
                    "minimum": {"os": "Windows 10"},
                    "recommended": {"os": "Windows 11"},
                }
            )
        )
        minimum = _record_for(
            evidence, evidence_by_field, "system_requirements.minimum.os"
        )
        recommended = _record_for(
            evidence, evidence_by_field, "system_requirements.recommended.os"
        )
        assert minimum.source_field != recommended.source_field

    def test_the_field_path_still_binds_to_the_draft_field(self) -> None:
        _, evidence, evidence_by_field = _map(
            _payload(
                name="X",
                release_date={"date": "Mar 25, 2022"},
                platforms={"windows": True},
                pc_requirements={"minimum": {"os": "Windows 10"}},
            )
        )
        for path, evidence_id in evidence_by_field.items():
            record = next(
                item for item in evidence if item.evidence_id == evidence_id
            )
            assert record.field_path == path

    def test_a_missing_field_gets_no_evidence(self) -> None:
        _, evidence, evidence_by_field = _map(
            _payload(name="X", release_date={"date": "Mar 25, 2022"})
        )
        assert "release_date" in evidence_by_field
        assert "platforms" not in evidence_by_field
        assert "system_requirements.minimum.os" not in evidence_by_field

    def test_a_missing_recommended_tier_gets_no_evidence(self) -> None:
        _, evidence, evidence_by_field = _map(
            _payload(pc_requirements={"minimum": {"os": "Windows 10"}})
        )
        assert "system_requirements.minimum.os" in evidence_by_field
        assert "system_requirements.recommended.os" not in evidence_by_field

    def test_evidence_ids_remain_unique(self) -> None:
        _, evidence, _ = _map(
            _payload(
                name="X",
                release_date={"date": "Mar 25, 2022"},
                platforms={"windows": True, "mac": True},
                pc_requirements={
                    "minimum": {"os": "Windows 10", "memory": "8 GB RAM"},
                    "recommended": {"os": "Windows 10", "memory": "16 GB RAM"},
                },
            )
        )
        ids = [item.evidence_id for item in evidence]
        assert len(ids) == len(set(ids))

    def test_the_source_reference_is_still_the_appdetails_url(self) -> None:
        _, evidence, _ = _map(_payload(name="X"))
        for item in evidence:
            assert item.source_ref is not None
            assert "appdetails" in item.source_ref
            assert f"appids={APP_ID}" in item.source_ref

    def test_the_source_type_is_still_steam(self) -> None:
        _, evidence, _ = _map(_payload(name="X", release_date={"date": "Mar 25, 2022"}))
        assert all(item.source_type == "steam" for item in evidence)

    def test_a_truncated_value_is_marked_and_not_promoted(self) -> None:
        long_value = "A" * 600
        _, evidence, evidence_by_field = _map(
            _payload(pc_requirements={"minimum": {"os": long_value}})
        )
        record = _record_for(
            evidence, evidence_by_field, "system_requirements.minimum.os"
        )
        assert record.truncated is True


def _record_for(
    evidence: List[Any],
    evidence_by_field: Dict[str, str],
    path: str,
) -> Any:
    """Return the evidence record bound to ``path``."""

    evidence_id = evidence_by_field[path]
    return next(item for item in evidence if item.evidence_id == evidence_id)


# ── Compatibility ────────────────────────────────────────────────────────


class TestContractCompatibility:
    """The extraction must not change the shape of the contract."""

    def test_the_facts_keep_their_existing_attributes(self) -> None:
        facts, _, _ = _map(_payload(name="X"))
        for attribute in ("name", "release_date", "platforms", "minimum", "recommended"):
            assert hasattr(facts, attribute)

    def test_the_supported_platform_list_is_unchanged(self) -> None:
        # The list the contract models still holds only the
        # platforms Steam positively reported.
        facts, _, _ = _map(
            _payload(platforms={"windows": True, "mac": False, "linux": True})
        )
        assert facts.platforms == ["Windows", "Linux"]

    def test_an_empty_payload_still_maps_to_empty_facts(self) -> None:
        facts, evidence, evidence_by_field = _map(
            {str(APP_ID): {"success": True, "data": {"name": "X"}}}
        )
        assert facts.name == "X"
        assert evidence
        assert evidence_by_field["name"]

    def test_a_payload_without_data_still_maps_to_nothing(self) -> None:
        facts, evidence, evidence_by_field = _map({str(APP_ID): {"success": False}})
        assert facts.name is None
        assert evidence == []
        assert evidence_by_field == {}
