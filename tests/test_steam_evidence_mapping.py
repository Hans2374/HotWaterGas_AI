"""Tests for the Steam field mapping.

The mapping is where a retrieval layer either stays honest or starts
inventing. These tests pin exactly which Steam fields reach the
contract, and — just as importantly — which ones must never appear.

Every test is a pure function call against a sanitized fixture: no
network, no clock, no model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from app.retrieval.steam_mapper import (
    canonical_platform,
    map_appdetails,
    normalize_requirement_text,
    parse_steam_date,
    strip_html,
    titles_match,
)


FIXTURES = Path(__file__).parent / "fixtures" / "steam"
RETRIEVED_AT = "2026-01-01T00:00:00+00:00"
APP_ID = 1245620


def _load(name: str) -> Dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _map(payload: Dict[str, Any]) -> Any:
    facts, evidence, evidence_by_field = map_appdetails(
        APP_ID, payload, retrieved_at=RETRIEVED_AT
    )
    return facts, evidence, evidence_by_field


# ── HTML stripping ────────────────────────────────────────────────────────


class TestHtmlStripping:
    """Steam requirement strings are small HTML fragments."""

    def test_tags_are_removed(self) -> None:
        assert strip_html("Windows 10 64-bit<br><strong>(64-bit)</strong>") == (
            "Windows 10 64-bit (64-bit)"
        )

    def test_entities_are_decoded(self) -> None:
        assert strip_html("Requires&nbsp;Intel&nbsp;Core i5") == (
            "Requires Intel Core i5"
        )

    def test_whitespace_is_collapsed(self) -> None:
        assert strip_html("Windows   10\n  64-bit") == "Windows 10 64-bit"

    def test_empty_input_returns_none(self) -> None:
        assert strip_html("") is None
        assert strip_html("   ") is None
        assert strip_html("<br>") is None

    def test_non_string_returns_none(self) -> None:
        assert strip_html(None) is None
        assert strip_html(123) is None

    def test_requirement_normalisation_is_case_insensitive(self) -> None:
        assert normalize_requirement_text("Windows 10 64-BIT") == (
            normalize_requirement_text("windows 10 64-bit")
        )

    def test_requirement_normalisation_keeps_digits(self) -> None:
        # Digits are meaningful: 8 GB and 16 GB must never compare equal.
        assert normalize_requirement_text("8 GB RAM") != normalize_requirement_text(
            "16 GB RAM"
        )


# ── Release date ──────────────────────────────────────────────────────────


class TestReleaseDateParsing:
    """Steam's release date is a localised, human-readable string."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Feb 25, 2022", "2022-02-25"),
            ("Mar 25, 2022", "2022-03-25"),
            ("25 Feb, 2022", "2022-02-25"),
            ("Dec 1, 1999", "1999-12-01"),
            ("2022-02-25", "2022-02-25"),
        ],
    )
    def test_parsable_dates_become_iso(self, raw: str, expected: str) -> None:
        assert parse_steam_date(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "Coming soon",
            "To be announced",
            "TBD",
            "not a date",
            "",
            None,
            123,
        ],
    )
    def test_unparsable_dates_stay_unknown(self, raw: Any) -> None:
        assert parse_steam_date(raw) is None

    def test_full_fixture_date_is_parsed(self) -> None:
        facts, _, _ = _map(_load("appdetails_success_full.json"))
        assert facts.release_date == "2022-02-25"

    def test_coming_soon_date_stays_unknown(self) -> None:
        facts, _, _ = _map(_load("appdetails_success_minimal.json"))
        assert facts.release_date is None


# ── Platforms ─────────────────────────────────────────────────────────────


class TestPlatformMapping:
    """Only truthy platform keys are mapped."""

    def test_only_truthy_keys_are_mapped(self) -> None:
        facts, _, _ = _map(_load("appdetails_success_full.json"))
        assert facts.platforms == ["Windows"]

    def test_multiple_truthy_keys_are_mapped(self) -> None:
        facts, _, _ = _map(_load("appdetails_with_excluded_fields.json"))
        assert facts.platforms == ["Windows", "macOS"]

    def test_absent_platforms_produce_an_empty_list(self) -> None:
        facts, _, _ = _map(
            {str(APP_ID): {"success": True, "data": {"name": "X"}}}
        )
        assert facts.platforms == []

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Windows", "Windows"),
            ("windows", "Windows"),
            ("PC", "Windows"),
            ("Mac OS X", "macOS"),
            ("macOS", "macOS"),
            ("Linux", "Linux"),
            ("Nintendo Switch", None),
            ("", None),
        ],
    )
    def test_canonical_platform_mapping(self, raw: str, expected: Optional[str]) -> None:
        assert canonical_platform(raw) == expected


# ── Requirements ──────────────────────────────────────────────────────────


class TestRequirementMapping:
    """The two tiers are mapped independently."""

    def test_minimum_tier_is_mapped(self) -> None:
        facts, _, _ = _map(_load("appdetails_success_full.json"))
        assert facts.minimum.os == "Windows 10 64-bit"
        assert facts.minimum.processor == "Intel Core i5-4460"
        assert facts.minimum.memory == "12 GB RAM"
        assert facts.minimum.graphics == "GTX 1050 Ti"
        assert facts.minimum.storage == "60 GB"

    def test_recommended_tier_is_mapped(self) -> None:
        facts, _, _ = _map(_load("appdetails_success_full.json"))
        assert facts.recommended.processor == "Intel Core i7-6700K"
        assert facts.recommended.memory == "16 GB RAM"

    def test_recommended_tier_is_not_copied_from_minimum(self) -> None:
        # The fixture's minimum and recommended differ. A tier must
        # only ever carry what its own source key carried.
        facts, _, _ = _map(_load("appdetails_success_full.json"))
        assert facts.recommended.memory != facts.minimum.memory

    def test_html_in_requirements_is_stripped(self) -> None:
        facts, _, _ = _map(_load("appdetails_with_excluded_fields.json"))
        assert facts.minimum.os == "Windows 10 64-bit (64-bit)"
        assert facts.minimum.processor == "Requires Intel Core i5"
        assert facts.minimum.memory == "8 GB RAM"

    def test_missing_requirements_produce_empty_tiers(self) -> None:
        facts, _, _ = _map(_load("appdetails_success_minimal.json"))
        assert facts.minimum.os is None
        assert facts.recommended.os is None

    def test_absent_pc_requirements_produce_empty_tiers(self) -> None:
        facts, _, _ = _map(
            {str(APP_ID): {"success": True, "data": {"name": "X"}}}
        )
        assert facts.minimum.os is None
        assert facts.recommended.processor is None

    def test_mac_requirements_are_not_mapped_into_the_pc_tier(self) -> None:
        # The contract models one tier per field, not one tier per
        # operating system. Folding the macOS values into the shared
        # PC fields would attribute macOS-only requirements to a
        # Windows product, so they are excluded.
        facts, _, _ = _map(_load("appdetails_with_excluded_fields.json"))
        assert facts.minimum.os == "Windows 10 64-bit (64-bit)"
        assert "macOS 12" not in (facts.minimum.os or "")

    def test_linux_requirements_are_not_mapped_into_the_pc_tier(self) -> None:
        facts, _, _ = _map(_load("appdetails_with_excluded_fields.json"))
        assert "Ubuntu 22.04" not in (facts.minimum.os or "")

    def test_notes_field_is_never_populated(self) -> None:
        # Steam has no "notes" field, so the tier must not invent one.
        facts, _, _ = _map(_load("appdetails_success_full.json"))
        assert not hasattr(facts.minimum, "notes")


# ── Excluded fields ───────────────────────────────────────────────────────


class TestExcludedFields:
    """Only a defensible mapping is made; the rest never reaches the prompt."""

    def test_price_is_not_mapped(self) -> None:
        facts, evidence, _ = _map(_load("appdetails_with_excluded_fields.json"))
        rendered = json.dumps(
            {
                "facts": facts.__dict__,
                "evidence": [item.model_dump() for item in evidence],
            },
            default=str,
        )
        assert "599000" not in rendered
        assert "price_overview" not in rendered
        assert "is_free" not in rendered

    def test_packages_are_not_mapped(self) -> None:
        _, evidence, _ = _map(_load("appdetails_with_excluded_fields.json"))
        rendered = json.dumps([item.model_dump() for item in evidence], default=str)
        assert "package" not in rendered.casefold()
        assert "12345" not in rendered

    def test_external_urls_are_not_mapped(self) -> None:
        _, evidence, _ = _map(_load("appdetails_with_excluded_fields.json"))
        rendered = json.dumps([item.model_dump() for item in evidence], default=str)
        assert "evil" not in rendered
        assert "example.com" not in rendered
        assert "cdn.example.com" not in rendered

    def test_genres_are_not_mapped_to_categories(self) -> None:
        # Steam genres are not the application's own taxonomy, and the
        # The contract has no genre field, so nothing is mapped.
        facts, evidence, _ = _map(_load("appdetails_with_excluded_fields.json"))
        rendered = json.dumps(
            {
                "facts": facts.__dict__,
                "evidence": [item.model_dump() for item in evidence],
            },
            default=str,
        )
        assert "Action" not in rendered
        assert "Adventure" not in rendered

    def test_store_categories_are_not_mapped(self) -> None:
        # Steam's ``categories`` are storefront ids (Single-player,
        # Steam Achievements), not the app's own category taxonomy.
        _, evidence, _ = _map(_load("appdetails_with_excluded_fields.json"))
        rendered = json.dumps([item.model_dump() for item in evidence], default=str)
        assert "Single-player" not in rendered
        assert "Steam Achievements" not in rendered

    def test_unrelated_metadata_is_not_mapped(self) -> None:
        _, evidence, _ = _map(_load("appdetails_with_excluded_fields.json"))
        rendered = json.dumps([item.model_dump() for item in evidence], default=str)
        assert "metacritic" not in rendered.casefold()
        assert "content_descriptors" not in rendered
        assert "achievements" not in rendered.casefold()


# ── Evidence ──────────────────────────────────────────────────────────────


class TestEvidenceCreation:
    """The parser owns the evidence it emits."""

    def test_every_mapped_field_has_evidence(self) -> None:
        _, evidence, evidence_by_field = _map(_load("appdetails_success_full.json"))
        assert evidence_by_field["name"]
        assert evidence_by_field["release_date"]
        assert evidence_by_field["platforms"]
        assert evidence_by_field["system_requirements.minimum.os"]
        assert evidence_by_field["system_requirements.recommended.os"]

    def test_evidence_source_type_is_steam(self) -> None:
        _, evidence, _ = _map(_load("appdetails_success_full.json"))
        assert all(item.source_type == "steam" for item in evidence)

    def test_evidence_excerpts_are_bounded(self) -> None:
        _, evidence, _ = _map(_load("appdetails_success_full.json"))
        for item in evidence:
            assert item.excerpt is not None
            assert len(item.excerpt) <= 500

    def test_evidence_ids_are_service_generated_and_prefixed(self) -> None:
        _, evidence, _ = _map(_load("appdetails_success_full.json"))
        for item in evidence:
            assert item.evidence_id.startswith("steam:1245620:")

    def test_evidence_ids_differ_per_field(self) -> None:
        _, evidence, _ = _map(_load("appdetails_success_full.json"))
        ids = {item.evidence_id for item in evidence}
        assert len(ids) == len(evidence)

    def test_no_evidence_is_created_for_absent_fields(self) -> None:
        _, evidence, evidence_by_field = _map(_load("appdetails_success_minimal.json"))
        assert "system_requirements.minimum.os" not in evidence_by_field
        assert "release_date" not in evidence_by_field


# ── Identity comparison ───────────────────────────────────────────────────


class TestTitleMatching:
    """The identity policy is conservative on purpose."""

    def test_exact_match(self) -> None:
        matched, reason = titles_match("ELDEN RING", "ELDEN RING")
        assert matched is True
        assert reason == "exact_match"

    def test_case_difference_is_not_a_mismatch(self) -> None:
        matched, _ = titles_match("Elden Ring", "ELDEN RING")
        assert matched is True

    def test_surrounding_whitespace_is_not_a_mismatch(self) -> None:
        matched, _ = titles_match("  Elden Ring  ", "ELDEN RING")
        assert matched is True

    def test_punctuation_difference_is_not_a_mismatch(self) -> None:
        matched, _ = titles_match("Aurora's Echo", "AURORAS ECHO")
        assert matched is True

    def test_containment_match_is_rejected(self) -> None:
        # A storefront title that merely extends the requested one is a
        # *different* product. The old policy accepted this pair
        # because the shorter title was long enough; that is exactly
        # the defect that let "Dark Souls" match "Dark Souls II".
        matched, reason = titles_match("Elden Ring", "Elden Ring Shadow of the Erdtree")
        assert matched is False
        assert reason == "title_has_subtitle"

    def test_short_title_does_not_match_by_containment(self) -> None:
        # "Hades" must not be treated as "Hades II".
        matched, _ = titles_match("Hades", "Hades II")
        assert matched is False

    def test_clearly_different_titles_do_not_match(self) -> None:
        matched, reason = titles_match("ELDEN RING", "Half-Life 2")
        assert matched is False
        assert reason == "title_differs"

    def test_missing_retrieved_title_does_not_match(self) -> None:
        matched, reason = titles_match("ELDEN RING", None)
        assert matched is False
        assert reason == "steam_reported_no_title"

    def test_empty_retrieved_title_does_not_match(self) -> None:
        matched, reason = titles_match("ELDEN RING", "   ")
        assert matched is False
        assert reason == "steam_title_is_empty"


# ── Identity policy: the D-1 regression suite ─────────────────────────────
#
# The old policy accepted a match whenever one normalised title
# contained the other and the shorter one was at least eight characters
# long. That rule conflated a base title with its sequel, its edition
# and its subtitled release, so a wrong App ID could enrich a product
# with another game's requirements.
#
# Every pair below is checked in **both directions**: the policy must
# be symmetric, because which side happens to be the caller's name and
# which side is Steam's title is not a fact the policy may depend on.


#: Pairs that are different products and must never match. Each entry is
#: ``(base, sequel)``.
SEQUEL_PAIRS = [
    ("Dark Souls", "Dark Souls II"),
    ("Half-Life", "Half-Life 2"),
    ("StarCraft", "StarCraft II"),
    ("Resident Evil", "Resident Evil 4"),
    ("Diablo II", "Diablo III"),
    ("Star Wars", "Star Warship"),
    ("Hades", "Hades II"),
    ("Portal", "Portal I"),
    ("Elden Ring", "Elden Ring Shadow of the Erdtree"),
    ("Elden Ring", "Elden Ring Definitive Edition"),
    ("Dota 2", "Dota 2 Reborn"),
    ("Mass Effect", "Mass Effect 2"),
    ("Fallout", "Fallout 4"),
    ("Doom", "Doom Eternal"),
]


class TestSequelAndEditionTitlesAreNeverMatched:
    """A base title and its sequel are two different products."""

    @pytest.mark.parametrize("base,sequel", SEQUEL_PAIRS)
    def test_base_does_not_match_sequel(self, base: str, sequel: str) -> None:
        matched, _ = titles_match(base, sequel)
        assert matched is False, f"{base!r} must not match {sequel!r}"

    @pytest.mark.parametrize("base,sequel", SEQUEL_PAIRS)
    def test_sequel_does_not_match_base(self, base: str, sequel: str) -> None:
        # The reverse direction must be rejected too: containment was
        # symmetric, so the fix has to be symmetric as well.
        matched, _ = titles_match(sequel, base)
        assert matched is False, f"{sequel!r} must not match {base!r}"

    def test_roman_numeral_sequel_is_rejected(self) -> None:
        matched, reason = titles_match("Dark Souls", "Dark Souls II")
        assert matched is False
        assert reason == "sequel_or_edition_marker"

    def test_arabic_numeral_sequel_is_rejected(self) -> None:
        matched, reason = titles_match("Half-Life", "Half-Life 2")
        assert matched is False
        assert reason == "sequel_or_edition_marker"

    def test_roman_versus_roman_sequel_is_rejected(self) -> None:
        matched, reason = titles_match("Diablo II", "Diablo III")
        assert matched is False
        assert reason == "sequel_or_edition_marker"

    def test_substring_extension_is_rejected(self) -> None:
        # "Star Wars" is a prefix of "Star Warship" but they are
        # different products. The old policy accepted this pair.
        matched, reason = titles_match("Star Wars", "Star Warship")
        assert matched is False
        assert reason == "title_is_prefix_of_other"

    def test_substring_extension_is_rejected_in_reverse(self) -> None:
        matched, reason = titles_match("Star Warship", "Star Wars")
        assert matched is False
        assert reason == "title_is_prefix_of_other"

    def test_subtitle_is_rejected(self) -> None:
        matched, reason = titles_match(
            "Elden Ring", "Elden Ring Shadow of the Erdtree"
        )
        assert matched is False
        assert reason == "title_has_subtitle"

    def test_edition_word_is_rejected(self) -> None:
        matched, reason = titles_match(
            "Elden Ring", "Elden Ring Definitive Edition"
        )
        assert matched is False
        assert reason == "sequel_or_edition_marker"


class TestNormalisationIsPreserved:
    """Legitimate normalisation must keep working."""

    @pytest.mark.parametrize(
        "requested,retrieved",
        [
            ("ELDEN RING", "ELDEN RING"),
            ("Elden Ring", "ELDEN RING"),
            ("elden ring", "ELDEN RING"),
            ("  Elden Ring  ", "ELDEN RING"),
            ("Elden   Ring", "Elden Ring"),
            ("Elden-Ring", "Elden Ring"),
            ("Elden: Ring", "Elden Ring"),
            ("Aurora's Echo", "AURORAS ECHO"),
            ("Aurora’s Echo", "AURORAS ECHO"),
            ("Aurora's Echo!", "Aurora's Echo"),
        ],
    )
    def test_normalised_exact_matches_are_accepted(
        self, requested: str, retrieved: str
    ) -> None:
        matched, reason = titles_match(requested, retrieved)
        assert matched is True, f"{requested!r} vs {retrieved!r}"
        assert reason == "exact_match"

    def test_digits_are_preserved_by_normalisation(self) -> None:
        # "Half-Life" and "Half-Life 2" must not normalise to the same
        # string: the digit is the whole difference between them.
        matched, _ = titles_match("Half-Life", "Half-Life 2")
        assert matched is False


class TestDegenerateTitles:
    """Empty and malformed names fail closed."""

    def test_empty_requested_name_does_not_match(self) -> None:
        matched, reason = titles_match("", "ELDEN RING")
        assert matched is False
        assert reason == "requested_name_is_empty"

    def test_whitespace_requested_name_does_not_match(self) -> None:
        matched, reason = titles_match("   ", "ELDEN RING")
        assert matched is False
        assert reason == "requested_name_is_empty"

    def test_punctuation_only_requested_name_does_not_match(self) -> None:
        matched, reason = titles_match("!!!", "ELDEN RING")
        assert matched is False
        assert reason == "requested_name_is_empty"

    def test_punctuation_only_retrieved_name_does_not_match(self) -> None:
        matched, reason = titles_match("ELDEN RING", "!!!")
        assert matched is False
        assert reason == "steam_title_is_empty"

    def test_none_retrieved_name_does_not_match(self) -> None:
        matched, reason = titles_match("ELDEN RING", None)
        assert matched is False
        assert reason == "steam_reported_no_title"

    def test_two_empty_titles_do_not_match(self) -> None:
        # Two empty names are not evidence of an identity.
        matched, reason = titles_match("", "")
        assert matched is False
        assert reason == "requested_name_is_empty"

    def test_single_character_title_matches_only_itself(self) -> None:
        matched, _ = titles_match("A", "A")
        assert matched is True
        matched, _ = titles_match("A", "AB")
        assert matched is False

    def test_numeric_titles_match_only_under_normalisation(self) -> None:
        matched, _ = titles_match("007", "007")
        assert matched is True
        matched, _ = titles_match("007", "008")
        assert matched is False


class TestIdentityPolicyIsSymmetric:
    """The verdict must not depend on which title came from where."""

    @pytest.mark.parametrize("left,right", SEQUEL_PAIRS)
    def test_verdict_is_the_same_in_both_directions(
        self, left: str, right: str
    ) -> None:
        forward = titles_match(left, right)
        backward = titles_match(right, left)
        assert forward[0] == backward[0]


# ── Release-year suffixes ─────────────────────────────────────────────────
#
# Steam appends a release year to disambiguate re-releases of the same
# game. The store title of the 2007 *Call of Duty 4: Modern Warfare*
# is "Call of Duty® 4: Modern Warfare® (2007)".
#
# The year is a **disambiguation aid, not a sequel marker**. Before the
# fix, the year was normalised into a bare "2007" token and the marker
# classifier read it as a numeric edition marker, so the identity check
# rejected a correct App ID with "sequel_or_edition_marker".
#
# The year is stripped from the **raw** title, before normalisation.
# That ordering is what keeps a release year apart from a franchise
# index: after normalisation "Portal 2" and "Portal (2002)" are the
# same bare token, and only the raw form can tell them apart.

#: The reported defect: App ID 7940, the 2007 *Call of Duty 4*.
COD4_REQUESTED = "Call of Duty 4: Modern Warfare"
COD4_STEAM_TITLE = "Call of Duty® 4: Modern Warfare® (2007)"


class TestReleaseYearSuffixIsNotASequelMarker:
    """A trailing release year must not be read as a sequel/edition."""

    def test_the_reported_false_positive_now_matches(self) -> None:
        matched, reason = titles_match(COD4_REQUESTED, COD4_STEAM_TITLE)
        assert matched is True
        assert reason == "exact_match"

    def test_the_reported_false_positive_matches_in_reverse(self) -> None:
        matched, _ = titles_match(COD4_STEAM_TITLE, COD4_REQUESTED)
        assert matched is True

    def test_a_genuinely_different_game_still_fails(self) -> None:
        # The 2022 *Call of Duty: Modern Warfare 4* is a different
        # product. The year handling must not turn it into a match.
        matched, reason = titles_match(
            "Call of Duty: Modern Warfare 4", COD4_STEAM_TITLE
        )
        assert matched is False
        assert reason == "title_differs"

    def test_a_genuinely_different_game_still_fails_in_reverse(self) -> None:
        matched, reason = titles_match(
            COD4_STEAM_TITLE, "Call of Duty: Modern Warfare 4"
        )
        assert matched is False
        assert reason == "title_differs"

    def test_bare_title_matches_the_year_suffixed_title(self) -> None:
        matched, reason = titles_match("Example Game", "Example Game (2020)")
        assert matched is True
        assert reason == "exact_match"

    def test_year_suffixed_title_matches_the_bare_title(self) -> None:
        matched, _ = titles_match("Example Game (2020)", "Example Game")
        assert matched is True

    def test_requested_year_is_stripped_too(self) -> None:
        # The caller's own product name may carry the year.
        matched, reason = titles_match("Doom (1999)", "Doom")
        assert matched is True
        assert reason == "exact_match"

    def test_both_sides_carrying_the_same_year_match(self) -> None:
        matched, reason = titles_match("Doom (1999)", "Doom (1999)")
        assert matched is True
        assert reason == "exact_match"

    def test_both_sides_carrying_different_years_do_not_match(self) -> None:
        matched, reason = titles_match("Doom (1999)", "Doom (2016)")
        assert matched is False
        assert reason == "release_year_differs"

    def test_different_years_do_not_match_in_reverse(self) -> None:
        matched, reason = titles_match("Doom (2016)", "Doom (1999)")
        assert matched is False
        assert reason == "release_year_differs"

    def test_a_year_does_not_hide_a_sequel(self) -> None:
        # The year is removed for the comparison, but the franchise
        # index in front of it is not. "Example Game 2 (2020)" is the
        # sequel of "Example Game", not the same game.
        matched, reason = titles_match("Example Game", "Example Game 2 (2020)")
        assert matched is False
        assert reason == "sequel_or_edition_marker"

    def test_a_year_does_not_hide_an_edition(self) -> None:
        # "Example Game (2020) Remastered" is a different product from
        # "Example Game": the year is not the trailing token, so it is
        # not a release-year suffix at all, and the remaster marker
        # still decides the comparison.
        matched, reason = titles_match(
            "Example Game", "Example Game (2020) Remastered"
        )
        assert matched is False
        assert reason == "sequel_or_edition_marker"

    def test_a_year_does_not_hide_a_sequel_number(self) -> None:
        # "Half-Life 2 (1998)" is Half-Life 2, not Half-Life.
        matched, reason = titles_match("Half-Life", "Half-Life 2 (1998)")
        assert matched is False
        assert reason == "sequel_or_edition_marker"

    def test_the_franchise_index_is_never_stripped(self) -> None:
        # The counterpart of the defect: a franchise index is not a
        # release year and must keep its sequel rejection.
        matched, reason = titles_match("Portal", "Portal 2")
        assert matched is False
        assert reason == "sequel_or_edition_marker"

    def test_a_year_is_not_stripped_from_the_middle_of_a_title(self) -> None:
        # "Fahrenheit 451" is a name, not a year suffix.
        matched, _ = titles_match("Fahrenheit 451", "Fahrenheit 451")
        assert matched is True

    def test_a_year_in_the_middle_is_never_stripped(self) -> None:
        # Only the *trailing* year is removed. The "451" in the middle
        # stays part of the name, so it still separates two products.
        matched, _ = titles_match("Fahrenheit 451", "Fahrenheit 452 (2020)")
        assert matched is False
        matched, _ = titles_match("Fahrenheit 451 (2020)", "Fahrenheit 452")
        assert matched is False

    def test_a_bracket_suffixed_year_is_recognised(self) -> None:
        matched, reason = titles_match("Example Game", "Example Game [2020]")
        assert matched is True
        assert reason == "exact_match"

    def test_the_verdict_is_symmetric_for_the_reported_case(self) -> None:
        forward = titles_match(COD4_REQUESTED, COD4_STEAM_TITLE)
        backward = titles_match(COD4_STEAM_TITLE, COD4_REQUESTED)
        assert forward[0] == backward[0] is True


class TestReleaseYearExtractionIsNarrow:
    """Only a plausible trailing year may be treated as a release year."""

    def test_a_year_below_the_range_is_not_a_release_year(self) -> None:
        # "0999" is out of the audited 1000-2999 range, so it stays in
        # the title and is read as a franchise index.
        matched, reason = titles_match("Example Game", "Example Game (0999)")
        assert matched is False
        assert reason == "sequel_or_edition_marker"

    def test_a_year_above_the_range_is_not_a_release_year(self) -> None:
        # "3000" is out of range.
        matched, reason = titles_match("Example Game", "Example Game (3000)")
        assert matched is False
        assert reason == "sequel_or_edition_marker"

    def test_the_lower_range_bound_is_inclusive(self) -> None:
        matched, reason = titles_match("Example Game", "Example Game (1000)")
        assert matched is True
        assert reason == "exact_match"

    def test_the_upper_range_bound_is_inclusive(self) -> None:
        matched, reason = titles_match("Example Game", "Example Game (2999)")
        assert matched is True
        assert reason == "exact_match"

    def test_a_title_that_is_only_a_year_does_not_match(self) -> None:
        # There is no base title to compare, so the year is not
        # stripped and the pair stays a mismatch.
        matched, _ = titles_match("Example Game", "(2020)")
        assert matched is False

    def test_a_three_digit_suffix_is_not_a_release_year(self) -> None:
        # "007" is a franchise index, not a year.
        matched, reason = titles_match("Example Game", "Example Game (007)")
        assert matched is False
        assert reason == "sequel_or_edition_marker"

    def test_a_five_digit_suffix_is_not_a_release_year(self) -> None:
        matched, reason = titles_match("Example Game", "Example Game (20201)")
        assert matched is False
        assert reason == "sequel_or_edition_marker"

    def test_an_unparenthesised_trailing_year_is_not_stripped(self) -> None:
        # Only a parenthesised or bracketed suffix is a release year.
        # A bare "2020" is a franchise index, like "Portal 2".
        matched, reason = titles_match("Example Game", "Example Game 2020")
        assert matched is False
        assert reason == "sequel_or_edition_marker"


class TestReleaseYearKeepsTheExistingProtections:
    """The existing sequel/edition/subtitle rejections must survive."""

    @pytest.mark.parametrize("base,sequel", SEQUEL_PAIRS)
    def test_every_existing_sequel_pair_still_fails(self, base: str, sequel: str) -> None:
        matched, _ = titles_match(base, sequel)
        assert matched is False, f"{base!r} must not match {sequel!r}"

    @pytest.mark.parametrize("base,sequel", SEQUEL_PAIRS)
    def test_every_existing_sequel_pair_still_fails_in_reverse(
        self, base: str, sequel: str
    ) -> None:
        matched, _ = titles_match(sequel, base)
        assert matched is False, f"{sequel!r} must not match {base!r}"

    @pytest.mark.parametrize("base,sequel", SEQUEL_PAIRS)
    def test_a_year_suffix_does_not_reopen_a_sequel_pair(
        self, base: str, sequel: str
    ) -> None:
        # A sequel that also carries a release year is still a sequel.
        matched, _ = titles_match(base, f"{sequel} (2020)")
        assert matched is False, f"{base!r} must not match {sequel!r} (2020)"

    @pytest.mark.parametrize(
        "requested,retrieved",
        [
            ("ELDEN RING", "ELDEN RING"),
            ("Elden Ring", "ELDEN RING"),
            ("elden ring", "ELDEN RING"),
            ("  Elden Ring  ", "ELDEN RING"),
            ("Elden   Ring", "Elden Ring"),
            ("Elden-Ring", "Elden Ring"),
            ("Elden: Ring", "Elden Ring"),
            ("Aurora's Echo", "AURORAS ECHO"),
            ("Aurora’s Echo", "AURORAS ECHO"),
            ("Aurora's Echo!", "Aurora's Echo"),
        ],
    )
    def test_normalisation_still_works(self, requested: str, retrieved: str) -> None:
        matched, reason = titles_match(requested, retrieved)
        assert matched is True, f"{requested!r} vs {retrieved!r}"
        assert reason == "exact_match"

    def test_digits_are_still_preserved_by_normalisation(self) -> None:
        matched, _ = titles_match("Half-Life", "Half-Life 2")
        assert matched is False


class TestDegenerateTitlesWithAReleaseYear:
    """Degenerate input stays safely rejected."""

    def test_empty_requested_name_does_not_match(self) -> None:
        matched, reason = titles_match("", "Example Game (2020)")
        assert matched is False
        assert reason == "requested_name_is_empty"

    def test_whitespace_requested_name_does_not_match(self) -> None:
        matched, reason = titles_match("   ", "Example Game (2020)")
        assert matched is False
        assert reason == "requested_name_is_empty"

    def test_empty_retrieved_name_does_not_match(self) -> None:
        matched, reason = titles_match("Example Game", "   ")
        assert matched is False
        assert reason == "steam_title_is_empty"

    def test_none_retrieved_name_does_not_match(self) -> None:
        matched, reason = titles_match("Example Game", None)
        assert matched is False
        assert reason == "steam_reported_no_title"

    def test_two_empty_titles_do_not_match(self) -> None:
        matched, reason = titles_match("", "")
        assert matched is False
        assert reason == "requested_name_is_empty"

    def test_a_year_only_requested_name_does_not_match(self) -> None:
        matched, _ = titles_match("(2020)", "Example Game")
        assert matched is False

    def test_the_verdict_is_symmetric_for_degenerate_input(self) -> None:
        forward = titles_match("", "Example Game (2020)")
        backward = titles_match("Example Game (2020)", "")
        assert forward[0] == backward[0] is False


class TestReleaseYearDoesNotAlterTheEvidence:
    """The retrieved Steam title must reach the facts unchanged."""

    def _cod4_payload(self) -> Dict[str, Any]:
        return {
            "7940": {
                "success": True,
                "data": {
                    "type": "game",
                    "name": COD4_STEAM_TITLE,
                    "short_description": "Call of Duty 4: Modern Warfare.",
                    "release_date": {"coming_soon": False, "date": "Nov 9, 2007"},
                    "platforms": {"windows": True, "mac": False, "linux": False},
                    "pc_requirements": {
                        "minimum": {
                            "os": "Windows XP",
                            "processor": "Intel Pentium 4 3.0GHz",
                            "memory": "1 GB RAM",
                            "graphics": "DirectX 9.0c capable",
                            "storage": "8 GB",
                        }
                    },
                },
            }
        }

    def test_the_facts_keep_the_original_steam_title(self) -> None:
        facts, _, _ = map_appdetails(
            7940, self._cod4_payload(), retrieved_at=RETRIEVED_AT
        )
        assert facts.name == COD4_STEAM_TITLE
        assert "(2007)" in facts.name

    def test_the_evidence_excerpt_keeps_the_original_steam_title(self) -> None:
        _, evidence, evidence_by_field = map_appdetails(
            7940, self._cod4_payload(), retrieved_at=RETRIEVED_AT
        )
        assert evidence_by_field["name"]
        name_evidence = next(
            item for item in evidence if item.evidence_id == evidence_by_field["name"]
        )
        assert name_evidence.excerpt == COD4_STEAM_TITLE
        assert "(2007)" in name_evidence.excerpt

    def test_the_evidence_source_reference_is_unchanged(self) -> None:
        _, evidence, evidence_by_field = map_appdetails(
            7940, self._cod4_payload(), retrieved_at=RETRIEVED_AT
        )
        name_evidence = next(
            item for item in evidence if item.evidence_id == evidence_by_field["name"]
        )
        assert "appids=7940" in name_evidence.source_ref

