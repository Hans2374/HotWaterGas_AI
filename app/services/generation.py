"""English→Vietnamese description translation seam.

This module owns the translation pipeline end to end: it retrieves
the English Steam description, selects and parses the source, builds a
translation-only prompt, calls the provider once and folds the
deterministic Steam facts into the response.

What this module does
---------------------
* Retrieves the **English** Steam description for the asserted App ID,
  through the same guarded retriever infrastructure the service uses
  (allow-listed host, bounded response size, timeouts, content-type
  check, no retries). The payload is fetched **once** and both the
  description source and the deterministic facts are read from it.
* Selects and parses the description source: headings, paragraphs and
  lists are preserved, repeated blocks are removed conservatively, and
  the source is cut at whole-block boundaries when it exceeds the
  source budget.
* Builds a **translation-only** prompt. The model's only job is to
  translate.
* Calls the provider exactly once and parses the structured payload.
* Folds the **deterministic** Steam facts into the response: the
  release date, the platform availability and the **Windows/PC**
  system requirements. macOS and Linux requirements are never included.
* Emits **warnings** for the conditions an admin needs to know about:
  a truncated source, an insufficient source, a translation that came
  out below the target range. Every warning is a statement about a
  well-defined thing: the truncation warnings describe the **input**,
  ``source_insufficient`` describes the **source**, and
  ``translation_below_target`` describes the **output**.
* Measures the translated text on the server. The model's own word
  count is discarded.

What this module deliberately does NOT do
----------------------------------------
* It does not mark the translation ``verified``. A translation that
  completed is not a fact that was checked, and the contract refuses a
  payload that claims otherwise.
* It does not invent text to reach the length target. When the source is
  too short, the service returns a shorter faithful translation **and**
  an admin-facing warning.
* It does not confuse a short translation with a short source. The
  source-sufficiency warning is derived from the source blocks alone;
  the output-length warning is derived from the translation alone.
* It does not silently truncate. A source that exceeds the budget is
  cut at whole-block boundaries and reported; a translation that
  exceeds the hard ceiling fails the call rather than being shortened.
* It does not merge macOS or Linux system requirements into the PC
  tiers.
"""

from __future__ import annotations

import json as _json
import logging
import time
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import ValidationError

from app.config import Settings
from app.errors import (
    GenerationDisabledError,
    InvalidProviderOutputError,
    MissingOllamaConfigError,
    ProviderError,
    RequestTooLargeError,
    SteamIdentityMismatchError,
    SteamNotFoundError,
    SteamRetrievalError,
)
from app.providers.base import StructuredProvider
from app.retrieval.description_source import (
    DescriptionSelection,
    SourceSufficiency,
    assess_source_sufficiency,
    select_description_source_from_payload,
)
from app.retrieval.steam_mapper import (
    STEAM_LOCALE_ENGLISH,
    map_appdetails,
    titles_match,
)
from app.schemas.description_fields import (
    EvidenceReference,
    FieldVerification,
    Platforms,
    ReleaseDate,
    RequirementField,
    RequirementTier,
    SystemRequirements,
)
from app.schemas.description import (
    SCHEMA_VERSION,
    DescriptionSourceBlock,
    DescriptionTranslationDraft,
    GenerateDescriptionRequest,
    GenerateDescriptionResponse,
    ModelMetadata,
    TranslatedDescription,
    DescriptionWarning,
)
from app.services.prompts import build_prompts
from app.validators.output import count_words


logger = logging.getLogger(__name__)


#: Version of the prompt template. Bump it whenever the prompt text
#: changes so a stored translation can be traced back to the prompt that
#: produced it.
PROMPT_VERSION = "v3.1"

#: The JSON Schema sent to the provider. The provider receives exactly
#: this document, so it doubles as the contract the model is held to.
PROVIDER_JSON_SCHEMA: Dict[str, Any] = DescriptionTranslationDraft.model_json_schema()

#: The requirement fields the response populates from the PC tier.
#: The ``notes`` field is excluded: Steam never publishes one.
_REQUIREMENT_FIELDS = ("os", "processor", "memory", "graphics", "storage")


@runtime_checkable
class SteamPayloadRetrieverLike(Protocol):
    """The narrow contract the service needs from a retriever.

    The service depends on this protocol rather than on the concrete
    :class:`~app.retrieval.steam.SteamAppIdRetriever`, so the retrieval
    can be replaced or faked in tests without a network.
    """

    def fetch_payload(self, app_id: Any) -> Any:
        """Retrieve and return the raw ``appdetails`` payload."""
        ...


class DescriptionTranslationService:
    """Orchestrates a single English→Vietnamese translation.

    The service is short-lived: it reads its configuration and provider
    at construction time. The lifespan handler builds one instance per
    app and stores it on ``app.state``.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        provider: StructuredProvider,
        steam_retriever: Optional["SteamPayloadRetrieverLike"] = None,
    ) -> None:
        self._settings = settings
        self._provider = provider
        self._steam_retriever = steam_retriever

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def model_id(self) -> str:
        """The model id the underlying provider will use."""
        return self._provider.model_id

    def attach_steam_retriever(
        self, retriever: Optional["SteamPayloadRetrieverLike"]
    ) -> None:
        """Attach (or detach) the optional Steam retriever.

        The retriever is attached after construction so the service can
        be built without any Steam configuration at all, and so a test
        can inject a fake without a network.
        """

        self._steam_retriever = retriever

    def generate(
        self, request: GenerateDescriptionRequest
    ) -> GenerateDescriptionResponse:
        """Produce a translated description for the given request.

        Raises a domain exception on failure. A failure is **never**
        returned as a partial response: the two are kept strictly apart.
        """

        if not self._settings.ai_generation_enabled:
            raise GenerationDisabledError(
                "AI generation is currently disabled on this server."
            )

        if not self._settings.ollama_model_id or not self._settings.ollama_base_url:
            raise MissingOllamaConfigError(
                "The Ollama provider is not fully configured on this server."
            )

        warnings: List[DescriptionWarning] = []
        retrieved_at = _utc_now_iso()

        # ── Retrieval ─────────────────────────────────────────────────
        # The payload is fetched **once**. Both the description source
        # and the deterministic facts are read from the same payload, so
        # the service never issues a second request for the same app.
        payload: Any = None
        if request.steam_app_id is not None:
            if self._steam_retriever is None:
                raise SteamRetrievalError(
                    "Steam retrieval is not available on this server.",
                    outcome="disabled",
                )
            try:
                payload = self._steam_retriever.fetch_payload(request.steam_app_id)
            except SteamIdentityMismatchError:
                raise
            except SteamNotFoundError:
                logger.warning("steam_not_found app_id=%s", request.steam_app_id)
                raise
            except SteamRetrievalError:
                logger.warning(
                    "steam_retrieval_failed app_id=%s", request.steam_app_id
                )
                raise

        # ── Deterministic facts ───────────────────────────────────────
        # The deterministic fact mapping, applied to the same payload.
        # It is locale-independent, and it maps the **PC** tier only:
        # Steam's macOS and Linux requirements are never read.
        facts = None
        evidence: List[EvidenceReference] = []
        evidence_by_field: Dict[str, str] = {}

        if payload is not None:
            facts, evidence, evidence_by_field = map_appdetails(
                request.steam_app_id or 0,
                payload,
                retrieved_at=retrieved_at,
                locale=STEAM_LOCALE_ENGLISH,
                extra_requirement_labels=True,
            )
            matched, reason = titles_match(request.product_name, facts.name)
            if not matched:
                logger.warning(
                    "steam_identity_mismatch app_id=%s reason=%s",
                    request.steam_app_id,
                    reason,
                )
                raise SteamIdentityMismatchError(
                    "The Steam App ID identifies a different game than the "
                    "supplied product name."
                )

        # ── Source selection ──────────────────────────────────────────
        selection = select_description_source_from_payload(
            payload,
            request.steam_app_id,
            max_source_bytes=self._settings.max_source_bytes,
        )

        if not selection.has_source:
            # There is nothing to translate. The call fails: producing an
            # empty description would be worse than saying so. This is a
            # controlled HTTP error, not a warning-bearing success, so
            # no warning code describes it.
            raise SteamRetrievalError(
                "The Steam app did not publish a translatable description.",
                outcome="no_description",
            )

        # The sufficiency assessment is made on the **source**, before
        # the model is called and without ever looking at its output.
        sufficiency = assess_source_sufficiency(
            selection,
            min_source_tokens=self._settings.min_source_tokens,
        )
        warnings.extend(_source_warnings(selection, sufficiency))

        # ── Prompt ────────────────────────────────────────────────────
        system_prompt, user_prompt = build_prompts(
            product_name=request.product_name,
            selection=selection,
            min_target_words=self._settings.min_target_words,
            max_target_words=self._settings.max_target_words,
        )

        _enforce_provider_input_size(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=PROVIDER_JSON_SCHEMA,
            limit=self._settings.max_provider_input_bytes,
        )

        # ── Provider ──────────────────────────────────────────────────
        started = time.monotonic()
        try:
            raw_payload = self._provider.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_schema=PROVIDER_JSON_SCHEMA,
            )
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "provider_contract_violation error_type=%s", type(exc).__name__
            )
            raise InvalidProviderOutputError(
                "The provider returned an unusable structured payload."
            ) from exc
        duration_ms = int((time.monotonic() - started) * 1000)

        draft = self._parse_draft(raw_payload)

        # ── Deterministic measurement ─────────────────────────────────
        text = draft.description.text
        word_count = count_words(text)
        if word_count > self._settings.max_description_words:
            raise InvalidProviderOutputError(
                f"provider output exceeds the "
                f"{self._settings.max_description_words}-word maximum"
            )
        within_range = (
            self._settings.min_target_words
            <= word_count
            <= self._settings.max_target_words
        )
        below_target = word_count < self._settings.min_target_words

        description = TranslatedDescription(
            text=text,
            language="vi",
            word_count=word_count,
            within_target_range=within_range,
            verification=draft.description.verification,
            evidence_ids=[],
        )

        # The soft target is advisory. A translation that came out
        # shorter than the target is *reported*, never padded.
        #
        # This is an **output** observation. It says nothing about the
        # source: a model may compress a perfectly adequate source into
        # fewer tokens, and a model may expand a thin source into more
        # tokens than the source really supported. The source-side
        # assessment lives in `sufficiency`, and the two are reported
        # under separate codes so a consumer can tell them apart.
        if below_target:
            warnings.append(
                DescriptionWarning(
                    code="translation_below_target",
                    severity="info",
                    message=(
                        f"The translation is {word_count} whitespace-separated "
                        f"tokens, below the soft target range of "
                        f"{self._settings.min_target_words}-"
                        f"{self._settings.max_target_words}. The text is a "
                        "faithful translation of the supplied source; it was "
                        "not padded, shortened or rejected."
                    ),
                    field_path="description",
                )
            )

        # ── Deterministic Steam facts ─────────────────────────────────
        release_date = ReleaseDate()
        platforms = Platforms()
        requirements = SystemRequirements()

        if facts is not None:
            release_date = ReleaseDate(
                value=facts.release_date,
                verification=_extracted_status(facts.release_date is not None),
                evidence_ids=_ids_for(evidence_by_field, "release_date"),
            )
            platforms = Platforms(
                values=list(facts.platforms),
                verification=_extracted_status(bool(facts.platforms)),
                evidence_ids=_ids_for(evidence_by_field, "platforms"),
            )

            minimum = RequirementTier()
            recommended = RequirementTier()
            for tier_name, tier in (
                ("minimum", facts.minimum),
                ("recommended", facts.recommended),
            ):
                target = minimum if tier_name == "minimum" else recommended
                for field_name in _REQUIREMENT_FIELDS:
                    value = getattr(tier, field_name, None)
                    if value is None:
                        continue
                    path = f"system_requirements.{tier_name}.{field_name}"
                    setattr(
                        target,
                        field_name,
                        RequirementField(
                            value=value,
                            verification=FieldVerification(status="unverified"),
                            evidence_ids=_ids_for(evidence_by_field, path),
                        ),
                    )
            requirements = SystemRequirements(
                minimum=minimum, recommended=recommended
            )

        # ── Response ──────────────────────────────────────────────────
        source_block = DescriptionSourceBlock(
            primary_field=selection.primary_field,
            supplementary_fields=list(selection.supplementary_fields),
            skipped_fields=list(selection.skipped_fields),
            block_count=len(selection.blocks),
            source_bytes=selection.used_bytes,
            total_source_bytes=selection.total_bytes,
            truncated=selection.truncated,
            dropped_blocks=selection.dropped_blocks,
            source_token_count=sufficiency.source_tokens,
            source_sufficient=sufficiency.sufficient,
            min_source_tokens=sufficiency.min_source_tokens,
        )

        return GenerateDescriptionResponse(
            schema_version=SCHEMA_VERSION,
            description=description,
            release_date=release_date,
            platforms=platforms,
            system_requirements=requirements,
            description_source=source_block,
            warnings=warnings,
            evidence=evidence,
            model=ModelMetadata(
                provider="ollama",
                model_id=self._provider.model_id,
                prompt_version=PROMPT_VERSION,
                duration_ms=duration_ms,
                generated_at=_utc_now_iso(),
            ),
            retrieved_at=retrieved_at,
        )

    # ── Parsing ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_draft(raw_payload: Dict[str, Any]) -> DescriptionTranslationDraft:
        """Validate the raw provider payload against the contract."""

        try:
            return DescriptionTranslationDraft.model_validate(raw_payload)
        except ValidationError as exc:
            logger.warning("schema_violation error_count=%s", exc.error_count())
            raise InvalidProviderOutputError(
                "The provider returned a payload that violates the schema."
            ) from exc


# ── Helpers ───────────────────────────────────────────────────────────────


def _source_warnings(
    selection: DescriptionSelection,
    sufficiency: SourceSufficiency,
) -> List[DescriptionWarning]:
    """Return the warnings the description source produced.

    Every warning here is a statement about the **input**, and none of
    them depends on what the model returned. The output-length
    observation lives in its own warning, emitted by the service after
    the translation has been measured.
    """

    warnings: List[DescriptionWarning] = []
    if selection.truncated:
        warnings.append(
            DescriptionWarning(
                code="source_truncated",
                severity="warning",
                message=(
                    f"The Steam description was longer than the source "
                    f"budget: {selection.dropped_blocks} block(s) were not "
                    "translated. The translation covers only the beginning "
                    "of the Steam description."
                ),
                field_path="description_source",
            )
        )
        if selection.dropped_blocks:
            warnings.append(
                DescriptionWarning(
                    code="source_block_dropped",
                    severity="warning",
                    message=(
                        f"{selection.dropped_blocks} source block(s) did not "
                        "fit inside the source budget and were dropped whole."
                    ),
                    field_path="description_source",
                )
            )

    if not sufficiency.sufficient:
        # The source, not the output, is what is being reported here.
        # The measurement is deterministic and is made on the selected
        # source blocks, before the model was called.
        warnings.append(
            DescriptionWarning(
                code="source_insufficient",
                severity="info",
                message=(
                    f"The English source carries {sufficiency.source_tokens} "
                    f"whitespace-separated tokens, below the "
                    f"{sufficiency.min_source_tokens} tokens the target range "
                    f"needs, and {sufficiency.reason}. The translation is "
                    "faithful to the material that was available; it was "
                    "not padded."
                ),
                field_path="description_source",
            )
        )
    return warnings


def _extracted_status(known: bool) -> FieldVerification:
    """Return the verification status for a deterministically extracted fact.

    The service does **not** run a deterministic verification pass
    over the model output: that pass was bound to the removed structured
    -draft model, and re-implementing it here would duplicate the rule in
    two places. The deterministic
    Steam facts are therefore reported as ``unverified`` - they were
    *extracted*, not checked against a second source.

    The one thing that is never allowed is a value with no status: a
    value that is present always carries ``unverified``, and a value
    that is absent carries ``unknown``.
    """

    if known:
        return FieldVerification(status="unverified")
    return FieldVerification(status="unknown")


def _ids_for(evidence_by_field: Dict[str, str], field_path: str) -> List[str]:
    """Return the evidence ids backing one field, if any."""

    evidence_id = evidence_by_field.get(field_path)
    return [evidence_id] if evidence_id else []


def _enforce_provider_input_size(
    *,
    system_prompt: str,
    user_prompt: str,
    json_schema: Dict[str, Any],
    limit: int,
) -> None:
    """Refuse to send an over-sized prompt to the model.

    The check runs **before** the provider is called, so an over-sized
    prompt never reaches a model, and the prompt is never trimmed.
    """

    schema_text = _json.dumps(
        json_schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    size = (
        len(system_prompt.encode("utf-8"))
        + len(user_prompt.encode("utf-8"))
        + len(schema_text.encode("utf-8"))
    )
    if size > limit:
        raise RequestTooLargeError(
            "The provider input exceeds the configured size limit."
        )


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""

    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "PROMPT_VERSION",
    "PROVIDER_JSON_SCHEMA",
    "DescriptionTranslationService",
    "SteamPayloadRetrieverLike",
]
