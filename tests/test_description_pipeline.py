"""Tests: the English Steam description translation pipeline.

The suite is entirely **offline**. Every provider interaction is served
by a hand-written fake, and every Steam interaction is served by a
sanitized fixture on disk. Nothing here reaches Ollama or Steam.

The tests are organised by the acceptance criteria of the stage:

* the real English Portal 2 fixture parses,
* the HTML structure survives (headings, paragraphs, lists),
* the description-source policy behaves (primary, fallback, supplement,
  supporting context, deduplication),
* the source budget truncates at whole-block boundaries and says so,
* the translation contract holds (target, ceiling, warnings, unverified prose),
* the deterministic facts are deterministic (date, platforms, Windows
  requirements only),
* the English locale is isolated from the default Vietnamese locale,
* the removed earlier routes stay gone.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from app.config import (
    DEFAULT_MAX_DESCRIPTION_WORDS,
    DEFAULT_MAX_SOURCE_BYTES,
    DEFAULT_MAX_TARGET_WORDS,
    DEFAULT_MIN_SOURCE_TOKENS,
    DEFAULT_MIN_TARGET_WORDS,
    Settings,
)
from app.errors import (
    InvalidProviderOutputError,
    RequestTooLargeError,
    SteamIdentityMismatchError,
    SteamNotFoundError,
    SteamRetrievalError,
)
from app.retrieval.description_source import (
    BLOCK_KIND_HEADING,
    BLOCK_KIND_LIST_ITEM,
    BLOCK_KIND_PARAGRAPH,
    ROLE_PRIMARY,
    ROLE_SUPPORTING,
    SOURCE_FIELD_ABOUT,
    SOURCE_FIELD_DETAILED,
    SOURCE_FIELD_SHORT,
    assess_source_sufficiency,
    parse_description_html,
    parse_description_text,
    select_description_source,
    select_description_source_from_payload,
)
from app.retrieval.steam import build_appdetails_url
from app.retrieval.steam_mapper import (
    STEAM_LOCALE,
    STEAM_LOCALE_ENGLISH,
    make_source_ref,
    map_appdetails,
)
from app.schemas.description import (
    SCHEMA_VERSION,
    WARNING_CODES,
    DescriptionSourceBlock,
    DescriptionTranslationDraft,
    GenerateDescriptionRequest,
    GenerateDescriptionResponse,
    TranslatedDescription,
    DescriptionWarning,
)
from app.services.generation import (
    PROVIDER_JSON_SCHEMA,
    PROMPT_VERSION,
    DescriptionTranslationService,
)
from app.services.prompts import build_prompts

FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "steam" / "appdetails_portal_2_english.json"
)

#: The App ID of the fixture app.
PORTAL_2_APP_ID = 620

#: A trailing version suffix on a generated definition name, e.g.
#: ``...V2`` or ``...V3``. The schema identities are version-neutral, so
#: a name that still carries a version suffix is a defect.
_VERSION_SUFFIX = re.compile(r"V\d+$")


def _registered_paths(app: Any) -> set[str]:
    """Return every route path the application exposes.

    FastAPI 0.140 replaces an included router with a wrapper object that
    does not carry a ``path`` attribute, so the paths are collected from
    the wrapped routers as well as from the plain routes. The OpenAPI
    document is used as a cross-check, because it is the public contract.
    """

    paths: set[str] = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        if isinstance(path, str):
            paths.add(path)
            continue
        original = getattr(route, "original_router", None)
        if original is None:
            continue
        for nested in getattr(original, "routes", ()):
            nested_path = getattr(nested, "path", None)
            if isinstance(nested_path, str):
                paths.add(nested_path)
    paths.update(app.openapi().get("paths", {}))
    return paths


# ── Fakes ─────────────────────────────────────────────────────────────────


class FakeProvider:
    """A structured provider that returns a canned payload.

    The fake records the prompts it received, so a test can assert what
    the model was shown without a network or a model.
    """

    def __init__(self, payload: Optional[Dict[str, Any]] = None) -> None:
        self._payload = payload if payload is not None else {
            "description": {"text": "Một bản dịch tiếng Việt."}
        }
        self.calls: List[Dict[str, Any]] = []

    @property
    def model_id(self) -> str:
        return "fake-model"

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "json_schema": json_schema,
            }
        )
        return self._payload


class FakeSteamRetriever:
    """A Steam retriever that serves a fixture instead of the network."""
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload
        self.payload_calls: List[Any] = []

    def fetch_payload(self, app_id: Any) -> Any:
        self.payload_calls.append(app_id)
        return self._payload

    def fetch(self, app_id: Any, *, product_name: str, retrieved_at: str) -> Any:
        from app.retrieval.steam_mapper import map_appdetails
        from app.retrieval.steam_models import (
            IdentityCheck,
            SteamRetrievalResult,
        )

        facts, evidence, evidence_by_field = map_appdetails(
            int(app_id),
            self._payload,
            retrieved_at=retrieved_at,
            locale=STEAM_LOCALE_ENGLISH,
        )
        from app.retrieval.steam_mapper import titles_match

        matched, reason = titles_match(product_name, facts.name)
        return SteamRetrievalResult(
            app_id=int(app_id),
            facts=facts,
            identity=IdentityCheck(
                matched=matched,
                requested_name=product_name,
                retrieved_name=facts.name,
                reason=reason,
            ),
            evidence=evidence,
            evidence_by_field=evidence_by_field,
        )


# ── Fixtures ──────────────────────────────────────────────────────────────


class _FakeTranslationService:
    """A translation service stand-in that returns a canned response."""

    def __init__(self) -> None:
        self.calls: List[Any] = []

    def generate(self, request: GenerateDescriptionRequest) -> GenerateDescriptionResponse:
        self.calls.append(request)
        return GenerateDescriptionResponse(
            description=TranslatedDescription(
                text="Một bản dịch tiếng Việt.", word_count=5
            ),
            model={
                "provider": "ollama",
                "model_id": "fake-model",
                "prompt_version": PROMPT_VERSION,
            },
        )

@pytest.fixture(scope="module")
def portal_2_payload() -> Dict[str, Any]:
    """The sanitized English Portal 2 App Details payload."""

    with FIXTURE_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture
def settings() -> Settings:
    """Settings that enable generation and Steam retrieval."""

    return Settings(
        ai_generation_enabled=True,
        llm_provider="ollama",
        internal_service_token="test-token",
        steam_retrieval_enabled=True,
    )


@pytest.fixture
def service(settings: Settings, portal_2_payload: Dict[str, Any]) -> DescriptionTranslationService:
    """A translation service wired to the fake provider and the fixture."""

    svc = DescriptionTranslationService(
        settings=settings,
        provider=FakeProvider(),
        steam_retriever=FakeSteamRetriever(portal_2_payload),
    )
    return svc


# ── The fixture itself ────────────────────────────────────────────────────


class TestFixture:
    """The sanitized fixture is what the tests assume it is."""

    def test_fixture_exists_and_is_valid_json(self) -> None:
        assert FIXTURE_PATH.exists()
        with FIXTURE_PATH.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        assert "620" in payload
        assert payload["620"]["success"] is True
        assert isinstance(payload["620"]["data"], dict)

    def test_fixture_retains_only_the_documented_fields(self) -> None:
        with FIXTURE_PATH.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        data = payload["620"]["data"]
        assert set(data.keys()) == {
            "type",
            "name",
            "detailed_description",
            "about_the_game",
            "short_description",
            "release_date",
            "platforms",
            "pc_requirements",
        }

    def test_fixture_excludes_pricing_and_user_content(self) -> None:
        with FIXTURE_PATH.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        data = payload["620"]["data"]
        for excluded in (
            "price_overview",
            "packages",
            "package_groups",
            "screenshots",
            "movies",
            "recommendations",
            "ratings",
            "achievements",
            "dlc",
            "website",
            "support_info",
            "required_age",
            "background",
            "header_image",
            "capsule_image",
            "background_raw",
            "capsule_imagev5",
            "mac_requirements",
            "linux_requirements",
        ):
            assert excluded not in data

    def test_fixture_preserves_the_source_html(self) -> None:
        with FIXTURE_PATH.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        data = payload["620"]["data"]
        # The markup is preserved verbatim, not rewritten.
        assert "<br>" in data["detailed_description"]
        assert "<ul" in data["detailed_description"]
        assert "<li>" in data["detailed_description"]
        assert "<strong>" in data["detailed_description"]
        assert "<ul" in data["pc_requirements"]["minimum"]
        assert "<li>" in data["pc_requirements"]["minimum"]

    def test_fixture_release_date_and_platforms(self) -> None:
        with FIXTURE_PATH.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        data = payload["620"]["data"]
        assert data["release_date"]["date"] == "18 Apr, 2011"
        assert data["platforms"] == {"windows": True, "mac": False, "linux": True}


# ── HTML parsing ──────────────────────────────────────────────────────────


class TestHtmlParsing:
    """The dedicated parser preserves the structure of the source."""

    def test_headings_are_preserved(self) -> None:
        blocks = parse_description_html("<h1>Title</h1><h3>Sub</h3>")
        assert [(b.kind, b.level, b.text) for b in blocks] == [
            (BLOCK_KIND_HEADING, 1, "Title"),
            (BLOCK_KIND_HEADING, 3, "Sub"),
        ]

    def test_paragraphs_are_preserved(self) -> None:
        blocks = parse_description_html("<p>First.</p><p>Second.</p>")
        assert [(b.kind, b.text) for b in blocks] == [
            (BLOCK_KIND_PARAGRAPH, "First."),
            (BLOCK_KIND_PARAGRAPH, "Second."),
        ]

    def test_unordered_list_items_are_preserved(self) -> None:
        blocks = parse_description_html("<ul><li>One</li><li>Two</li></ul>")
        assert [b.kind for b in blocks] == [
            BLOCK_KIND_LIST_ITEM,
            BLOCK_KIND_LIST_ITEM,
        ]
        assert [b.text for b in blocks] == ["One", "Two"]
        assert all(b.ordered is False for b in blocks)

    def test_ordered_list_items_are_marked_ordered(self) -> None:
        blocks = parse_description_html("<ol><li>One</li><li>Two</li></ol>")
        assert all(b.ordered is True for b in blocks)
        assert blocks[0].render() == "1. One"

    def test_line_breaks_split_blocks(self) -> None:
        blocks = parse_description_html("Alpha<br><br>Beta")
        assert [b.text for b in blocks] == ["Alpha", "Beta"]

    def test_inline_markup_contributes_text_without_a_block(self) -> None:
        blocks = parse_description_html("<p>Hello <strong>world</strong></p>")
        assert len(blocks) == 1
        assert blocks[0].text == "Hello world"
        assert blocks[0].kind == BLOCK_KIND_PARAGRAPH

    def test_standalone_bold_text_becomes_a_heading(self) -> None:
        blocks = parse_description_html("<strong>Product Features</strong><br>")
        assert blocks[0].kind == BLOCK_KIND_HEADING
        assert blocks[0].text == "Product Features"

    def test_inline_bold_phrase_stays_a_paragraph(self) -> None:
        blocks = parse_description_html("<p>Some <strong>bold</strong> words</p>")
        assert blocks[0].kind == BLOCK_KIND_PARAGRAPH

    def test_entities_are_decoded(self) -> None:
        blocks = parse_description_html("<p>a &amp; b &quot;c&quot;</p>")
        assert blocks[0].text == 'a & b "c"'

    def test_missing_description_yields_no_blocks(self) -> None:
        assert parse_description_html(None) == []
        assert parse_description_html("") == []
        assert parse_description_html("   ") == []
        assert parse_description_html(123) == []

    def test_plain_text_becomes_a_single_paragraph(self) -> None:
        blocks = parse_description_text("A short summary.")
        assert len(blocks) == 1
        assert blocks[0].kind == BLOCK_KIND_PARAGRAPH
        assert blocks[0].text == "A short summary."

    def test_plain_text_decodes_entities(self) -> None:
        blocks = parse_description_text("The &quot;Initiative&quot; grew.")
        assert blocks[0].text == 'The "Initiative" grew.'

    def test_empty_plain_text_yields_no_blocks(self) -> None:
        assert parse_description_text(None) == []
        assert parse_description_text("  ") == []


class TestFixtureParsing:
    """The real English Portal 2 description parses as expected."""

    def test_the_fixture_yields_the_expected_structure(
        self, portal_2_payload: Dict[str, Any]
    ) -> None:
        selection = select_description_source_from_payload(portal_2_payload, PORTAL_2_APP_ID)
        kinds = [b.kind for b in selection.blocks]
        assert kinds.count(BLOCK_KIND_PARAGRAPH) >= 3
        assert kinds.count(BLOCK_KIND_LIST_ITEM) == 6
        assert BLOCK_KIND_HEADING in kinds

    def test_the_heading_is_the_product_features_title(
        self, portal_2_payload: Dict[str, Any]
    ) -> None:
        selection = select_description_source_from_payload(portal_2_payload, PORTAL_2_APP_ID)
        headings = [b.text for b in selection.blocks if b.kind == BLOCK_KIND_HEADING]
        assert headings == ["Product Features"]

    def test_the_list_items_keep_their_order(
        self, portal_2_payload: Dict[str, Any]
    ) -> None:
        selection = select_description_source_from_payload(portal_2_payload, PORTAL_2_APP_ID)
        items = [b.text for b in selection.blocks if b.kind == BLOCK_KIND_LIST_ITEM]
        assert items[0].startswith("Extensive single player:")
        assert items[-1].startswith("Editing Tools:")

    def test_the_detailed_description_is_the_primary_source(
        self, portal_2_payload: Dict[str, Any]
    ) -> None:
        selection = select_description_source_from_payload(portal_2_payload, PORTAL_2_APP_ID)
        assert selection.primary_field == SOURCE_FIELD_DETAILED
        assert selection.blocks[0].role == ROLE_PRIMARY

    def test_the_source_is_not_truncated_at_the_default_budget(
        self, portal_2_payload: Dict[str, Any]
    ) -> None:
        selection = select_description_source_from_payload(portal_2_payload, PORTAL_2_APP_ID)
        assert selection.truncated is False
        assert selection.dropped_blocks == 0
        assert selection.used_bytes <= DEFAULT_MAX_SOURCE_BYTES


# ── The source policy ─────────────────────────────────────────────────────


class TestSourcePolicy:
    """Selection, fallback, supplement, supporting context, dedup."""

    def test_detailed_description_is_preferred_over_about_the_game(self) -> None:
        selection = select_description_source(
            {
                "detailed_description": "<p>Primary body.</p>",
                "about_the_game": "<p>Different body.</p>",
            }
        )
        assert selection.primary_field == SOURCE_FIELD_DETAILED
        texts = [b.text for b in selection.blocks]
        assert "Primary body." in texts
        assert "Different body." in texts
        assert SOURCE_FIELD_ABOUT in selection.supplementary_fields

    def test_about_the_game_is_the_fallback_when_the_primary_is_absent(self) -> None:
        selection = select_description_source({"about_the_game": "<p>Only body.</p>"})
        assert selection.primary_field == SOURCE_FIELD_ABOUT
        assert [b.text for b in selection.blocks] == ["Only body."]
        assert selection.truncated is False

    def test_about_the_game_contributes_distinct_supplementary_content(self) -> None:
        selection = select_description_source(
            {
                "detailed_description": "<p>Primary body.</p>",
                "about_the_game": "<p>Primary body.</p><p>Extra body.</p>",
            }
        )
        texts = [b.text for b in selection.blocks]
        assert texts.count("Primary body.") == 1
        assert "Extra body." in texts
        assert SOURCE_FIELD_ABOUT in selection.supplementary_fields

    def test_about_the_game_is_skipped_when_fully_duplicated(self) -> None:
        selection = select_description_source(
            {
                "detailed_description": "<p>Same body.</p>",
                "about_the_game": "<p>Same body.</p>",
            }
        )
        assert SOURCE_FIELD_ABOUT in selection.skipped_fields
        assert SOURCE_FIELD_ABOUT not in selection.supplementary_fields
        assert [b.text for b in selection.blocks] == ["Same body."]

    def test_short_description_is_added_when_it_adds_context(self) -> None:
        selection = select_description_source(
            {
                "detailed_description": "<p>Long body.</p>",
                "short_description": "A brief summary.",
            }
        )
        texts = [b.text for b in selection.blocks]
        assert "A brief summary." in texts
        assert SOURCE_FIELD_SHORT in selection.supplementary_fields
        supporting = [b for b in selection.blocks if b.role == ROLE_SUPPORTING]
        assert len(supporting) == 1

    def test_short_description_is_not_appended_when_it_is_already_present(self) -> None:
        selection = select_description_source(
            {
                "detailed_description": "<p>A brief summary.</p>",
                "short_description": "A brief summary.",
            }
        )
        assert [b.text for b in selection.blocks] == ["A brief summary."]
        assert SOURCE_FIELD_SHORT in selection.skipped_fields

    def test_short_description_is_not_appended_when_contained_in_the_source(self) -> None:
        selection = select_description_source(
            {
                "detailed_description": (
                    "<p>The Perpetual Testing Initiative has been expanded to "
                    "allow you to design co-op puzzles.</p>"
                ),
                "short_description": "The Perpetual Testing Initiative has been expanded.",
            }
        )
        texts = [b.text for b in selection.blocks]
        assert texts == [
            "The Perpetual Testing Initiative has been expanded to allow you "
            "to design co-op puzzles."
        ]
        assert SOURCE_FIELD_SHORT in selection.skipped_fields

    def test_deduplication_is_conservative_and_keeps_distinct_content(self) -> None:
        selection = select_description_source(
            {
                "detailed_description": "<p>Same.</p><p>Different.</p>",
                "about_the_game": "<p>Same.</p><p>Also different.</p>",
            }
        )
        texts = [b.text for b in selection.blocks]
        assert texts.count("Same.") == 1
        assert "Different." in texts
        assert "Also different." in texts

    def test_deduplication_ignores_case_and_whitespace(self) -> None:
        selection = select_description_source(
            {
                "detailed_description": "<p>The   Same Body.</p>",
                "about_the_game": "<p>the same body</p>",
            }
        )
        assert [b.text for b in selection.blocks] == ["The Same Body."]
        assert SOURCE_FIELD_ABOUT in selection.skipped_fields

    def test_a_similarly_but_not_identically_worded_block_is_kept(self) -> None:
        selection = select_description_source(
            {
                "detailed_description": "<p>Portal 2 was released in 2011.</p>",
                "about_the_game": "<p>Portal 2 was released in 2012.</p>",
            }
        )
        texts = [b.text for b in selection.blocks]
        assert "Portal 2 was released in 2011." in texts
        assert "Portal 2 was released in 2012." in texts

    def test_missing_description_fields_produce_no_source(self) -> None:
        selection = select_description_source({})
        assert selection.has_source is False
        assert selection.primary_field is None
        assert selection.blocks == []

    def test_an_empty_description_produces_no_source(self) -> None:
        selection = select_description_source({"detailed_description": "   "})
        assert selection.has_source is False

    def test_the_selection_accepts_a_whole_envelope(self, portal_2_payload) -> None:
        selection = select_description_source(portal_2_payload)
        assert selection.has_source is True

    def test_the_selection_accepts_a_raw_data_object(self, portal_2_payload) -> None:
        selection = select_description_source(portal_2_payload["620"]["data"])
        assert selection.has_source is True

    def test_source_order_is_preserved(self) -> None:
        selection = select_description_source(
            {
                "detailed_description": "<p>One.</p><p>Two.</p>",
                "about_the_game": "<p>Three.</p>",
                "short_description": "Four.",
            }
        )
        texts = [b.text for b in selection.blocks]
        assert texts.index("One.") < texts.index("Two.") < texts.index("Three.")
        assert texts.index("Three.") < texts.index("Four.")


# ── The source budget ─────────────────────────────────────────────────────


class TestSourceBudget:
    """Truncation happens at whole-block boundaries, and says so."""

    def test_truncation_drops_whole_blocks(self) -> None:
        selection = select_description_source(
            {
                "detailed_description": "".join(
                    f"<p>Paragraph number {index} with some text.</p>"
                    for index in range(50)
                )
            },
            max_source_bytes=400,
        )
        assert selection.truncated is True
        assert selection.dropped_blocks > 0
        assert selection.used_bytes <= 400

    def test_truncation_preserves_the_source_order(self) -> None:
        selection = select_description_source(
            {
                "detailed_description": "".join(
                    f"<p>Paragraph number {index} with some text.</p>"
                    for index in range(50)
                )
            },
            max_source_bytes=400,
        )
        kept = [b.text for b in selection.blocks]
        assert kept == sorted(kept, key=lambda text: int(text.split()[2]))
        assert kept[0] == "Paragraph number 0 with some text."

    def test_truncation_never_splits_a_block(self) -> None:
        selection = select_description_source(
            {
                "detailed_description": "".join(
                    f"<p>Paragraph number {index} with some text.</p>"
                    for index in range(50)
                )
            },
            max_source_bytes=400,
        )
        for block in selection.blocks:
            assert block.text.startswith("Paragraph number ")
            assert block.text.endswith(" with some text.")

    def test_the_source_is_not_truncated_when_it_fits(self) -> None:
        selection = select_description_source(
            {"detailed_description": "<p>Short.</p>"},
            max_source_bytes=10_000,
        )
        assert selection.truncated is False
        assert selection.dropped_blocks == 0

    def test_a_single_oversized_block_is_kept_and_flagged(self) -> None:
        selection = select_description_source(
            {"detailed_description": "<p>" + ("x" * 500) + "</p>"},
            max_source_bytes=100,
        )
        assert selection.truncated is True
        assert len(selection.blocks) == 1

    def test_a_zero_budget_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            select_description_source({"detailed_description": "<p>x</p>"}, max_source_bytes=0)


# ── The source-sufficiency assessment ─────────────────────────────────────


class TestSourceSufficiency:
    """The source assessment is made on the source, never on the output."""

    def test_a_source_above_the_threshold_is_sufficient(self) -> None:
        selection = select_description_source(
            {"detailed_description": "<p>" + ("word " * 500) + "</p>"},
            max_source_bytes=100_000,
        )
        assessment = assess_source_sufficiency(selection, min_source_tokens=100)
        assert assessment.sufficient is True
        assert assessment.source_tokens == 500
        assert assessment.truncated is False

    def test_a_source_below_the_threshold_is_insufficient(self) -> None:
        selection = select_description_source(
            {"detailed_description": "<p>" + ("word " * 20) + "</p>"},
            max_source_bytes=100_000,
        )
        assessment = assess_source_sufficiency(selection, min_source_tokens=100)
        assert assessment.sufficient is False
        assert assessment.source_tokens == 20

    def test_the_threshold_boundary_is_inclusive(self) -> None:
        selection = select_description_source(
            {"detailed_description": "<p>" + ("word " * 100) + "</p>"},
            max_source_bytes=100_000,
        )
        assessment = assess_source_sufficiency(selection, min_source_tokens=100)
        assert assessment.sufficient is True

    def test_a_truncated_source_is_never_sufficient(self) -> None:
        # The blocks that fit would have been enough on their own, but
        # the dropped blocks are exactly the material that would have
        # carried the translation past the target.
        selection = select_description_source(
            {
                "detailed_description": "".join(
                    f"<p>Paragraph number {index} with some text.</p>"
                    for index in range(50)
                )
            },
            max_source_bytes=400,
        )
        assessment = assess_source_sufficiency(selection, min_source_tokens=1)
        assert selection.truncated is True
        assert assessment.sufficient is False
        assert assessment.truncated is True

    def test_the_assessment_never_sees_the_model_output(self) -> None:
        # The function takes only the selection: there is no parameter
        # through which a translation could enter the assessment.
        import inspect

        parameters = list(
            inspect.signature(assess_source_sufficiency).parameters.keys()
        )
        assert parameters == ["selection", "min_source_tokens"]

    def test_a_non_positive_threshold_is_rejected(self) -> None:
        selection = select_description_source(
            {"detailed_description": "<p>x</p>"}, max_source_bytes=100
        )
        with pytest.raises(ValueError):
            assess_source_sufficiency(selection, min_source_tokens=0)

    def test_the_documented_default_threshold_is_used(self) -> None:
        assert DEFAULT_MIN_SOURCE_TOKENS == 350

    def test_the_real_portal_2_source_is_insufficient(
        self, portal_2_payload: Dict[str, Any]
    ) -> None:
        # The genuine English Portal 2 description is short — roughly
        # 230 tokens — so the assessment must call it insufficient.
        # This is the honest answer, and it is why the warning is
        # computed from the source rather than from the model's output.
        selection = select_description_source_from_payload(portal_2_payload, 620)
        assessment = assess_source_sufficiency(
            selection, min_source_tokens=DEFAULT_MIN_SOURCE_TOKENS
        )
        assert assessment.sufficient is False
        assert assessment.source_tokens == 233
        assert assessment.source_tokens < DEFAULT_MIN_SOURCE_TOKENS

    def test_the_threshold_is_documented(self) -> None:
        assert DEFAULT_MIN_SOURCE_TOKENS == 350


# ── The translation contract ───────────────────────────────────────────────


class TestTranslationContract:
    """The response contract holds its rules."""

    def test_the_schema_version_is_three(self) -> None:
        assert SCHEMA_VERSION == "3"

    def test_a_translated_description_cannot_be_verified(self) -> None:
        with pytest.raises(Exception):
            TranslatedDescription(
                text="Bản dịch.",
                word_count=2,
                verification={"status": "verified"},
            )

    def test_a_translated_description_defaults_to_unverified(self) -> None:
        description = TranslatedDescription(text="Bản dịch.", word_count=2)
        assert description.verification.status == "unverified"

    def test_an_unknown_status_is_upgraded_to_unverified(self) -> None:
        description = TranslatedDescription(
            text="Bản dịch.", word_count=2, verification={"status": "unknown"}
        )
        assert description.verification.status == "unverified"

    def test_an_undocumented_warning_code_is_rejected(self) -> None:
        with pytest.raises(Exception):
            GenerateDescriptionResponse(
                description=TranslatedDescription(text="Bản dịch.", word_count=2),
                model={
                    "provider": "ollama",
                    "model_id": "fake",
                    "prompt_version": PROMPT_VERSION,
                },
                warnings=[{"code": "made_up_code", "message": "nope"}],
            )

    def test_the_documented_warning_codes_are_stable(self) -> None:
        assert "source_truncated" in WARNING_CODES
        assert "source_insufficient" in WARNING_CODES
        assert "translation_below_target" in WARNING_CODES
        assert "source_block_dropped" in WARNING_CODES
        assert "description_over_ceiling" in WARNING_CODES

    def test_the_unobservable_source_missing_code_is_gone(self) -> None:
        # The no-description path is a controlled HTTP error, so a
        # `source_missing` warning could never appear in a delivered
        # body. A code a consumer can never observe is a code a consumer
        # cannot rely on.
        assert "source_missing" not in WARNING_CODES

    def test_the_response_rejects_a_dangling_evidence_reference(self) -> None:
        with pytest.raises(Exception):
            GenerateDescriptionResponse(
                description=TranslatedDescription(text="Bản dịch.", word_count=2),
                model={
                    "provider": "ollama",
                    "model_id": "fake",
                    "prompt_version": PROMPT_VERSION,
                },
                release_date={"value": "2011-04-18", "evidence_ids": ["steam:620:nope"]},
            )

    def test_the_response_rejects_a_duplicated_evidence_id(self) -> None:
        record = {
            "evidence_id": "steam:620:release_date",
            "field_path": "release_date",
            "source_type": "steam",
        }
        with pytest.raises(Exception):
            GenerateDescriptionResponse(
                description=TranslatedDescription(text="Bản dịch.", word_count=2),
                model={
                    "provider": "ollama",
                    "model_id": "fake",
                    "prompt_version": PROMPT_VERSION,
                },
                evidence=[record, dict(record)],
            )

    def test_the_provider_schema_carries_only_the_description(self) -> None:
        assert "description" in PROVIDER_JSON_SCHEMA["properties"]
        assert "release_date" not in PROVIDER_JSON_SCHEMA["properties"]
        assert "platforms" not in PROVIDER_JSON_SCHEMA["properties"]

    def test_the_draft_model_ignores_unknown_keys(self) -> None:
        draft = DescriptionTranslationDraft.model_validate(
            {"description": {"text": "Bản dịch.", "word_count": 2}, "extra": 1}
        )
        assert draft.description.text == "Bản dịch."

    def test_the_description_source_block_defaults_to_empty(self) -> None:
        block = DescriptionSourceBlock()
        assert block.primary_field is None
        assert block.block_count == 0
        assert block.truncated is False


class TestTranslationRequest:
    """The request contract is strict."""

    def test_an_empty_product_name_is_rejected(self) -> None:
        with pytest.raises(Exception):
            GenerateDescriptionRequest(product_name="   ")

    def test_a_boolean_app_id_is_rejected(self) -> None:
        with pytest.raises(Exception):
            GenerateDescriptionRequest(product_name="Portal 2", steam_app_id=True)

    def test_a_float_app_id_is_rejected(self) -> None:
        with pytest.raises(Exception):
            GenerateDescriptionRequest(product_name="Portal 2", steam_app_id=620.5)

    def test_a_negative_app_id_is_rejected(self) -> None:
        with pytest.raises(Exception):
            GenerateDescriptionRequest(product_name="Portal 2", steam_app_id=-1)

    def test_a_numeric_string_app_id_is_accepted(self) -> None:
        request = GenerateDescriptionRequest(product_name="Portal 2", steam_app_id="620")
        assert request.steam_app_id == 620

    def test_unknown_fields_are_rejected(self) -> None:
        with pytest.raises(Exception):
            GenerateDescriptionRequest(product_name="Portal 2", surprise=1)


# ── The translation service ────────────────────────────────────────────────


class TestTranslationService:
    """The service orchestrates the translation end to end."""

    def _request(self) -> GenerateDescriptionRequest:
        return GenerateDescriptionRequest(
            product_name="Portal 2", steam_app_id=PORTAL_2_APP_ID
        )

    def test_a_successful_translation_returns_the_contract(
        self, service: DescriptionTranslationService
    ) -> None:
        response = service.generate(self._request())
        assert response.schema_version == "3"
        assert response.description.text == "Một bản dịch tiếng Việt."
        assert response.description.language == "vi"

    def test_the_description_stays_unverified(
        self, service: DescriptionTranslationService
    ) -> None:
        response = service.generate(self._request())
        assert response.description.verification.status == "unverified"
        assert response.description.verification.checked_at is None

    def test_the_model_cannot_mark_the_description_verified(
        self, settings: Settings, portal_2_payload: Dict[str, Any]
    ) -> None:
        provider = FakeProvider(
            {
                "description": {
                    "text": "Bản dịch.",
                    "word_count": 2,
                    "verification": {"status": "verified"},
                }
            }
        )
        svc = DescriptionTranslationService(
            settings=settings,
            provider=provider,
            steam_retriever=FakeSteamRetriever(portal_2_payload),
        )
        with pytest.raises(InvalidProviderOutputError):
            svc.generate(self._request())

    def test_the_word_count_is_measured_on_the_server(
        self, settings: Settings, portal_2_payload: Dict[str, Any]
    ) -> None:
        provider = FakeProvider(
            {"description": {"text": "một hai ba bốn", "word_count": 999}}
        )
        svc = DescriptionTranslationService(
            settings=settings,
            provider=provider,
            steam_retriever=FakeSteamRetriever(portal_2_payload),
        )
        response = svc.generate(self._request())
        assert response.description.word_count == 4
        assert response.description.within_target_range is False

    def test_a_short_translation_is_reported_as_below_target(
        self, service: DescriptionTranslationService
    ) -> None:
        # The Portal 2 source is genuinely short, so BOTH warnings fire:
        # one about the source, one about the output.
        response = service.generate(self._request())
        codes = [w.code for w in response.warnings]
        assert "translation_below_target" in codes
        below = [w for w in response.warnings if w.code == "translation_below_target"]
        assert below[0].severity == "info"

    def test_a_short_translation_alone_does_not_claim_an_insufficient_source(
        self, settings: Settings, portal_2_payload: Dict[str, Any]
    ) -> None:
        # The source is made long enough to be sufficient, and the model
        # returns a short answer. The output warning fires; the source
        # warning must NOT, because the source was fine.
        settings = settings.model_copy(update={"min_source_tokens": 100})
        provider = FakeProvider({"description": {"text": "một hai ba"}})
        svc = DescriptionTranslationService(
            settings=settings,
            provider=provider,
            steam_retriever=FakeSteamRetriever(portal_2_payload),
        )
        response = svc.generate(self._request())
        codes = [w.code for w in response.warnings]
        assert "translation_below_target" in codes
        assert "source_insufficient" not in codes
        assert response.description_source.source_sufficient is True
        assert response.description.within_target_range is False

    def test_a_sufficient_source_with_a_long_translation_is_clean(
        self, settings: Settings, portal_2_payload: Dict[str, Any]
    ) -> None:
        settings = settings.model_copy(update={"min_source_tokens": 100})
        provider = FakeProvider(
            {
                "description": {
                    "text": " ".join(["từ"] * DEFAULT_MIN_TARGET_WORDS)
                }
            }
        )
        svc = DescriptionTranslationService(
            settings=settings,
            provider=provider,
            steam_retriever=FakeSteamRetriever(portal_2_payload),
        )
        response = svc.generate(self._request())
        codes = [w.code for w in response.warnings]
        assert "translation_below_target" not in codes
        assert "source_insufficient" not in codes
        assert response.description.within_target_range is True
        assert response.description_source.source_sufficient is True

    def test_a_translation_over_the_hard_ceiling_fails(
        self, settings: Settings, portal_2_payload: Dict[str, Any]
    ) -> None:
        provider = FakeProvider(
            {"description": {"text": " ".join(["từ"] * (DEFAULT_MAX_DESCRIPTION_WORDS + 1))}}
        )
        svc = DescriptionTranslationService(
            settings=settings,
            provider=provider,
            steam_retriever=FakeSteamRetriever(portal_2_payload),
        )
        with pytest.raises(InvalidProviderOutputError):
            svc.generate(self._request())

    def test_a_translation_at_the_hard_ceiling_is_accepted(
        self, settings: Settings, portal_2_payload: Dict[str, Any]
    ) -> None:
        provider = FakeProvider(
            {"description": {"text": " ".join(["từ"] * DEFAULT_MAX_DESCRIPTION_WORDS)}}
        )
        svc = DescriptionTranslationService(
            settings=settings,
            provider=provider,
            steam_retriever=FakeSteamRetriever(portal_2_payload),
        )
        response = svc.generate(self._request())
        assert response.description.word_count == DEFAULT_MAX_DESCRIPTION_WORDS

    def test_a_truncated_source_is_reported(
        self, settings: Settings, portal_2_payload: Dict[str, Any]
    ) -> None:
        settings = settings.model_copy(
            update={"max_source_bytes": 400}
        )
        svc = DescriptionTranslationService(
            settings=settings,
            provider=FakeProvider(),
            steam_retriever=FakeSteamRetriever(portal_2_payload),
        )
        response = svc.generate(self._request())
        codes = [w.code for w in response.warnings]
        assert "source_truncated" in codes
        assert "source_block_dropped" in codes
        assert response.description_source.truncated is True
        assert response.description_source.dropped_blocks > 0

    def test_a_truncated_source_never_claims_the_whole_description(
        self, settings: Settings, portal_2_payload: Dict[str, Any]
    ) -> None:
        settings = settings.model_copy(update={"max_source_bytes": 400})
        svc = DescriptionTranslationService(
            settings=settings,
            provider=FakeProvider(),
            steam_retriever=FakeSteamRetriever(portal_2_payload),
        )
        response = svc.generate(self._request())
        assert response.description_source.block_count < 11
        assert response.description_source.total_source_bytes > (
            response.description_source.source_bytes
        )

    def test_the_provider_receives_the_whole_selected_source(
        self, settings: Settings, portal_2_payload: Dict[str, Any]
    ) -> None:
        provider = FakeProvider()
        svc = DescriptionTranslationService(
            settings=settings,
            provider=provider,
            steam_retriever=FakeSteamRetriever(portal_2_payload),
        )
        svc.generate(self._request())
        call = provider.calls[0]
        assert "Portal 2 draws from the award-winning formula" in call["user_prompt"]
        assert "Product Features" in call["user_prompt"]
        assert "Editing Tools:" in call["user_prompt"]

    def test_the_source_is_fetched_exactly_once(
        self, settings: Settings, portal_2_payload: Dict[str, Any]
    ) -> None:
        retriever = FakeSteamRetriever(portal_2_payload)
        svc = DescriptionTranslationService(
            settings=settings,
            provider=FakeProvider(),
            steam_retriever=retriever,
        )
        svc.generate(self._request())
        assert retriever.payload_calls == [PORTAL_2_APP_ID]

    def test_a_missing_retriever_fails_the_call(
        self, settings: Settings
    ) -> None:
        svc = DescriptionTranslationService(
            settings=settings, provider=FakeProvider(), steam_retriever=None
        )
        with pytest.raises(SteamRetrievalError):
            svc.generate(self._request())

    def test_an_app_with_no_description_fails_the_call(
        self, settings: Settings
    ) -> None:
        payload = {"620": {"success": True, "data": {"name": "Portal 2"}}}
        svc = DescriptionTranslationService(
            settings=settings,
            provider=FakeProvider(),
            steam_retriever=FakeSteamRetriever(payload),
        )
        with pytest.raises(SteamRetrievalError):
            svc.generate(self._request())

    def test_an_identity_mismatch_fails_the_call(
        self, settings: Settings, portal_2_payload: Dict[str, Any]
    ) -> None:
        svc = DescriptionTranslationService(
            settings=settings,
            provider=FakeProvider(),
            steam_retriever=FakeSteamRetriever(portal_2_payload),
        )
        request = GenerateDescriptionRequest(
            product_name="Completely Different Game", steam_app_id=PORTAL_2_APP_ID
        )
        with pytest.raises(SteamIdentityMismatchError):
            svc.generate(request)

    def test_a_provider_contract_violation_fails_the_call(
        self, settings: Settings, portal_2_payload: Dict[str, Any]
    ) -> None:
        provider = FakeProvider({"description": {"text": ""}})
        svc = DescriptionTranslationService(
            settings=settings,
            provider=provider,
            steam_retriever=FakeSteamRetriever(portal_2_payload),
        )
        with pytest.raises(InvalidProviderOutputError):
            svc.generate(self._request())

    def test_the_soft_target_is_not_enforced_as_a_requirement(
        self, service: DescriptionTranslationService
    ) -> None:
        response = service.generate(self._request())
        assert response.description.word_count < DEFAULT_MIN_TARGET_WORDS
        assert response.description.within_target_range is False


# ── The deterministic facts ───────────────────────────────────────────────


class TestDeterministicFacts:
    """The release date, the platforms and the requirements are deterministic."""

    def _response(self, service: DescriptionTranslationService):
        return service.generate(
            GenerateDescriptionRequest(
                product_name="Portal 2", steam_app_id=PORTAL_2_APP_ID
            )
        )

    def test_the_release_date_is_parsed_from_the_steam_json(
        self, service: DescriptionTranslationService
    ) -> None:
        response = self._response(service)
        assert response.release_date.value == "2011-04-18"
        assert response.release_date.verification.status == "unverified"

    def test_the_release_date_carries_evidence(
        self, service: DescriptionTranslationService
    ) -> None:
        response = self._response(service)
        assert response.release_date.evidence_ids
        assert response.release_date.evidence_ids[0] in {
            item.evidence_id for item in response.evidence
        }

    def test_the_platform_availability_uses_the_steam_booleans(
        self, service: DescriptionTranslationService
    ) -> None:
        response = self._response(service)
        assert response.platforms.values == ["Windows", "Linux"]

    def test_the_minimum_requirements_are_the_windows_ones(
        self, service: DescriptionTranslationService
    ) -> None:
        response = self._response(service)
        minimum = response.system_requirements.minimum
        assert minimum.os.value == "Windows 7 / Vista / XP"
        assert minimum.processor.value is not None
        assert minimum.memory.value == "2 GB RAM"
        assert minimum.storage.value == "8 GB available space"

    def test_the_recommended_tier_is_not_derived_from_the_minimum(
        self, service: DescriptionTranslationService
    ) -> None:
        response = self._response(service)
        assert response.system_requirements.recommended.os.value is None
        assert response.system_requirements.recommended.memory.value is None

    def test_no_macos_or_linux_requirements_are_present(
        self, service: DescriptionTranslationService
    ) -> None:
        response = self._response(service)
        for tier_name in ("minimum", "recommended"):
            tier = getattr(response.system_requirements, tier_name)
            for field_name in ("os", "processor", "memory", "graphics", "storage"):
                value = getattr(tier, field_name).value
                if value is None:
                    continue
                lowered = value.casefold()
                assert "os x" not in lowered
                assert "macos" not in lowered
                assert "ubuntu" not in lowered

    def test_the_facts_are_never_marked_verified(
        self, service: DescriptionTranslationService
    ) -> None:
        response = self._response(service)
        assert response.release_date.verification.status != "verified"
        assert response.platforms.verification.status != "verified"
        for tier_name in ("minimum", "recommended"):
            tier = getattr(response.system_requirements, tier_name)
            for field_name in tier.fields():
                assert getattr(tier, field_name).verification.status != "verified"

    def test_the_evidence_names_the_english_source(
        self, service: DescriptionTranslationService
    ) -> None:
        response = self._response(service)
        for item in response.evidence:
            assert item.source_ref is not None
            assert f"l={STEAM_LOCALE_ENGLISH}" in item.source_ref

    def test_the_evidence_names_the_source_field(
        self, service: DescriptionTranslationService
    ) -> None:
        response = self._response(service)
        source_fields = {item.source_field for item in response.evidence}
        assert "release_date.date" in source_fields
        assert "pc_requirements.minimum" in source_fields
        # The platform record names the individual Steam keys it was
        # built from, comma separated.
        platform_fields = [
            field for field in source_fields if field and field.startswith("platforms.")
        ]
        assert platform_fields
        assert "platforms.windows" in platform_fields[0]


# ── Locale isolation ──────────────────────────────────────────────────────


class TestLocaleIsolation:
    """The English locale is separate from the mapper's default locale."""

    def test_the_default_locale_is_unchanged(self) -> None:
        # The mapper's default locale is a property of the retained
        # Steam infrastructure, not of any generation pipeline.
        assert STEAM_LOCALE == "vietnamese"

    def test_the_english_locale_is_a_separate_constant(self) -> None:
        assert STEAM_LOCALE_ENGLISH == "english"
        assert STEAM_LOCALE_ENGLISH != STEAM_LOCALE

    def test_the_default_request_url_uses_the_default_locale(self) -> None:
        url = build_appdetails_url(620)
        assert "l=vietnamese" in url
        assert "l=english" not in url

    def test_the_english_request_url_is_opt_in(self) -> None:
        url = build_appdetails_url(620, locale=STEAM_LOCALE_ENGLISH)
        assert "l=english" in url
        assert "l=vietnamese" not in url

    def test_the_default_source_reference_is_unchanged(self) -> None:
        assert make_source_ref(620) == (
            "https://store.steampowered.com/api/appdetails?appids=620"
            "&l=vietnamese&cc=vn"
        )

    def test_the_english_source_reference_names_the_english_locale(self) -> None:
        assert "l=english" in make_source_ref(620, locale=STEAM_LOCALE_ENGLISH)

    def test_the_default_mapping_still_records_the_default_locale(
        self, portal_2_payload: Dict[str, Any]
    ) -> None:
        _facts, evidence, _by_field = map_appdetails(
            PORTAL_2_APP_ID, portal_2_payload, retrieved_at="2026-11-02T00:00:00+00:00"
        )
        assert evidence
        for item in evidence:
            assert "l=vietnamese" in (item.source_ref or "")

    def test_the_english_mapping_records_the_english_locale(
        self, portal_2_payload: Dict[str, Any]
    ) -> None:
        _facts, evidence, _by_field = map_appdetails(
            PORTAL_2_APP_ID,
            portal_2_payload,
            retrieved_at="2026-11-02T00:00:00+00:00",
            locale=STEAM_LOCALE_ENGLISH,
        )
        assert evidence
        for item in evidence:
            assert "l=english" in (item.source_ref or "")

    def test_the_translation_bounds_are_separate(self) -> None:
        assert DEFAULT_MIN_TARGET_WORDS == 500
        assert DEFAULT_MAX_TARGET_WORDS == 600
        assert DEFAULT_MAX_DESCRIPTION_WORDS == 800


# ── The prompt ────────────────────────────────────────────────────────────


class TestPrompt:
    """The prompt is translation-only and carries the whole source."""

    def _prompts(self):
        selection = select_description_source(
            {"detailed_description": "<h2>Features</h2><p>Body.</p>"}
        )
        return build_prompts(
            product_name="Portal 2",
            selection=selection,
            min_target_words=DEFAULT_MIN_TARGET_WORDS,
            max_target_words=DEFAULT_MAX_TARGET_WORDS,
        )

    def test_the_system_prompt_forbids_invention(self) -> None:
        system_prompt, _user = self._prompts()
        assert "KHÔNG thêm" in system_prompt
        assert "KHÔNG dùng kiến thức ngoài" in system_prompt
        assert "KHÔNG tóm tắt bỏ mất" in system_prompt

    def test_the_system_prompt_states_the_soft_target(self) -> None:
        system_prompt, _user = self._prompts()
        assert str(DEFAULT_MIN_TARGET_WORDS) in system_prompt
        assert str(DEFAULT_MAX_TARGET_WORDS) in system_prompt
        assert "MỀM" in system_prompt

    def test_the_system_prompt_forbids_self_verification(self) -> None:
        system_prompt, _user = self._prompts()
        assert "unverified" in system_prompt

    def test_the_user_prompt_carries_the_whole_source(self) -> None:
        _system_prompt, user_prompt = self._prompts()
        assert "Features" in user_prompt
        assert "Body." in user_prompt

    def test_the_user_prompt_marks_the_source_as_data(self) -> None:
        _system_prompt, user_prompt = self._prompts()
        assert "không phải chỉ dẫn" in user_prompt

    def test_a_truncated_source_is_flagged_in_the_prompt(self) -> None:
        selection = select_description_source(
            {"detailed_description": "".join(
                f"<p>Block number {i} with distinct content here.</p>"
                for i in range(50)
            )},
            max_source_bytes=100,
        )
        _system_prompt, user_prompt = build_prompts(
            product_name="Portal 2",
            selection=selection,
            min_target_words=DEFAULT_MIN_TARGET_WORDS,
            max_target_words=DEFAULT_MAX_TARGET_WORDS,
        )
        assert "bị cắt" in user_prompt


# ── The provider-facing schema ────────────────────────────────────────────


class TestProviderSchemaStructure:
    """The provider-facing document must not drift.

    The JSON Schema is sent verbatim to the model provider, so its
    *structure* — the types, the bounds, the enumerations, the required
    sets and the reference graph — is part of the observable contract.

    The **labels** in that document (the root title and the ``$defs``
    keys) are the models' class names, and the rename was allowed to
    change them. They are therefore replaced by placeholders before the
    comparison; everything else is compared exactly.
    """

    GOLDEN = Path(__file__).parent / "fixtures" / "provider_schema_golden.json"

    @staticmethod
    def _labels(schema: dict) -> list[str]:
        """Return every label the document names."""

        labels = []
        title = schema.get("title")
        if isinstance(title, str):
            labels.append(title)
        labels.extend(schema.get("$defs", {}).keys())
        return labels

    @classmethod
    def _normalise(cls, schema: dict) -> dict:
        """Replace every label with a stable placeholder."""

        text = json.dumps(schema, sort_keys=True, ensure_ascii=False)
        # Longest first, so a label that is a prefix of another cannot
        # corrupt it.
        for index, name in enumerate(
            sorted(cls._labels(schema), key=len, reverse=True)
        ):
            text = re.sub(
                r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])",
                f"<LABEL_{index}>",
                text,
            )
        return json.loads(text)

    def test_the_golden_artifact_exists(self) -> None:
        # Without the artifact the comparison below would be vacuous.
        assert self.GOLDEN.is_file(), f"missing golden artifact: {self.GOLDEN}"

    def test_the_provider_schema_structure_has_not_drifted(self) -> None:
        from app.services.generation import PROVIDER_JSON_SCHEMA

        golden = json.loads(self.GOLDEN.read_text(encoding="utf-8"))
        assert self._normalise(PROVIDER_JSON_SCHEMA) == golden["structure"], (
            "the provider-facing JSON Schema has drifted from its golden "
            "structure; the provider would receive a different document"
        )

    def test_the_definition_names_are_the_neutral_class_names(self) -> None:
        from app.services.generation import PROVIDER_JSON_SCHEMA

        golden = json.loads(self.GOLDEN.read_text(encoding="utf-8"))
        assert sorted(PROVIDER_JSON_SCHEMA.get("$defs", {})) == golden[
            "definition_names"
        ]
        for name in golden["definition_names"]:
            assert not _VERSION_SUFFIX.search(name), (
                f"stale version suffix in {name!r}"
            )

    def test_every_model_forbids_extra_properties(self) -> None:
        # The strictness of the contract is part of its structure: a
        # model that accepted an unknown property would let the model
        # widen the contract.
        from app.services.generation import PROVIDER_JSON_SCHEMA

        defs = PROVIDER_JSON_SCHEMA.get("$defs", {})
        assert defs, "the provider schema has no definitions"
        for name, definition in defs.items():
            assert definition.get("additionalProperties") is False, name

    def test_the_required_sets_are_declared(self) -> None:
        from app.services.generation import PROVIDER_JSON_SCHEMA

        defs = PROVIDER_JSON_SCHEMA.get("$defs", {})
        for name, definition in defs.items():
            properties = definition.get("properties", {})
            required = definition.get("required", [])
            # A required set never names a property that does not exist.
            for field in required:
                assert field in properties, f"{name}.{field}"
            # And it never repeats a name.
            assert len(required) == len(set(required)), name


# ── The removed routes ────────────────────────────────────────────────────


class TestRegression:
    """The translation route is registered, and the removed earlier routes stay gone."""

    def test_the_retained_routes_are_still_registered(self) -> None:
        from app.main import create_app

        app = create_app()
        paths = _registered_paths(app)
        assert "/internal/descriptions/generate" in paths
        assert "/internal/steam/search" in paths
        assert "/health" in paths

    def test_the_legacy_alias_is_still_registered(self) -> None:
        # The path the route was published under before the rename is
        # kept as an alias, so a deployment that has not been updated
        # keeps working during the migration.
        from app.main import create_app

        app = create_app()
        paths = _registered_paths(app)
        assert "/internal/descriptions/generate-v3" in paths

    def test_the_removed_legacy_route_is_gone(self) -> None:
        # The legacy generation route was removed together with its
        # Gemini-backed pipeline. It must not reappear under any other
        # spelling.
        from app.main import create_app

        app = create_app()
        paths = _registered_paths(app)
        assert "/internal/descriptions/generate-description" not in paths

    def test_the_removed_structured_draft_route_is_gone(self) -> None:
        # The structured-draft route was removed together with its
        # pipeline. The translation route did not take its place, and it
        # must not reappear under the old path.
        from app.main import create_app

        app = create_app()
        paths = _registered_paths(app)
        assert "/internal/descriptions/generate-v2" not in paths

    def test_the_description_source_ref_uses_the_english_locale(self) -> None:
        assert "l=english" in make_source_ref(620, locale=STEAM_LOCALE_ENGLISH)

    def test_the_request_uses_the_english_locale_by_default(self) -> None:
        from app.retrieval.steam import SteamAppIdRetriever

        retriever = SteamAppIdRetriever(locale=STEAM_LOCALE_ENGLISH)
        assert retriever.locale == STEAM_LOCALE_ENGLISH

    def test_the_prompt_version_is_unchanged(self) -> None:
        from app.services.generation import PROMPT_VERSION

        assert PROMPT_VERSION == "v3.1"


# ── The route ─────────────────────────────────────────────────────────────


class TestTranslationRoute:
    """The translation endpoint is wired the way the contract says."""

    @pytest.fixture
    def client(self, monkeypatch: pytest.MonkeyPatch):
        """A TestClient with the translation service replaced by a fake."""

        from fastapi.testclient import TestClient

        monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "test-internal-token-12345")
        monkeypatch.setenv("AI_GENERATION_ENABLED", "true")

        with patch("app.main.build_structured_provider") as build_mock:
            build_mock.return_value = object()
            from app.main import create_app

            app = create_app()
            with TestClient(app) as client:
                app.state.translation_service = _FakeTranslationService()
                yield client

    def test_the_route_requires_the_internal_token(self, client) -> None:
        response = client.post(
            "/internal/descriptions/generate",
            json={"product_name": "Portal 2", "steam_app_id": 620},
        )
        assert response.status_code == 401
        assert response.json()["code"] == "unauthorized"

    def test_an_incorrect_token_returns_401(self, client) -> None:
        response = client.post(
            "/internal/descriptions/generate",
            headers={"X-Internal-Token": "wrong-token"},
            json={"product_name": "Portal 2", "steam_app_id": 620},
        )
        assert response.status_code == 401

    def test_the_route_rejects_an_invalid_payload(self, client) -> None:
        response = client.post(
            "/internal/descriptions/generate",
            headers={"X-Internal-Token": "test-internal-token-12345"},
            json={"product_name": "Portal 2", "steam_app_id": -5},
        )
        assert response.status_code == 422

    def test_a_successful_call_returns_the_contract(self, client) -> None:
        response = client.post(
            "/internal/descriptions/generate",
            headers={"X-Internal-Token": "test-internal-token-12345"},
            json={"product_name": "Portal 2", "steam_app_id": 620},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["schema_version"] == "3"
        assert body["description"]["verification"]["status"] == "unverified"
        assert "warnings" in body
        assert "description_source" in body

    def test_generation_disabled_is_reported_as_503(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fastapi.testclient import TestClient
        from app.main import create_app

        monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "test-internal-token-12345")
        with TestClient(create_app()) as client:
            response = client.post(
                "/internal/descriptions/generate",
                headers={"X-Internal-Token": "test-internal-token-12345"},
                json={"product_name": "Portal 2", "steam_app_id": 620},
            )
        assert response.status_code == 503
        assert response.json()["code"] == "generation_disabled"

    def test_the_openapi_contract_documents_the_route(self) -> None:
        from fastapi.testclient import TestClient
        from app.main import create_app

        with TestClient(create_app()) as client:
            schema = client.get("/openapi.json").json()
        assert "/internal/descriptions/generate" in schema["paths"]
        post = schema["paths"]["/internal/descriptions/generate"]["post"]
        for code in ("200", "401", "404", "409", "413", "422", "500", "502", "503", "504"):
            assert code in post["responses"], code

    def test_the_openapi_contract_does_not_document_the_legacy_alias(self) -> None:
        # The alias exists so an in-flight deployment keeps working. It
        # is deliberately excluded from the published contract: a
        # consumer must be able to tell which path is canonical.
        from fastapi.testclient import TestClient
        from app.main import create_app

        with TestClient(create_app()) as client:
            schema = client.get("/openapi.json").json()
        assert "/internal/descriptions/generate-v3" not in schema["paths"]

    def test_the_legacy_alias_still_answers(self, client) -> None:
        # The alias is served by the same handler, so it behaves
        # identically to the canonical path.
        response = client.post(
            "/internal/descriptions/generate-v3",
            headers={"X-Internal-Token": "test-internal-token-12345"},
            json={"product_name": "Portal 2", "steam_app_id": 620},
        )
        assert response.status_code == 200
        assert response.json()["schema_version"] == "3"
