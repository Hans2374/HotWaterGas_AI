"""Data structures produced by the Steam retrieval layer.

The retrieval layer returns **facts and evidence**, never a draft. The
service decides what to do with them. Keeping the two apart is what
makes the retrieval testable without a model and keeps the model's
output from ever being mistaken for a retrieved fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.schemas.description_fields import EvidenceReference


@dataclass(frozen=True)
class SteamPlatformAvailability:
    """Which operating systems Steam reports as supported.

    The three states are kept strictly apart, because they are three
    different facts:

    * ``True`` — Steam explicitly reports the platform as supported.
    * ``False`` — Steam explicitly reports the platform as **not**
      supported.
    * ``None`` — Steam did not say. A missing key, a ``null`` value and
      a value of an unexpected type all land here: "not reported" is
      not the same fact as "reported as unsupported", and it is
      certainly not the same as "supported".

    Platform availability is entirely separate from the system
    requirements: a game can be supported on macOS while the
    contract still carries only the PC/Windows requirement tiers.
    """

    windows: Optional[bool] = None
    mac: Optional[bool] = None
    linux: Optional[bool] = None


@dataclass(frozen=True)
class SteamRequirements:
    """One requirement tier as extracted from Steam.

    Every field is ``None`` when Steam did not list it. A tier is never
    derived from the other tier: ``recommended`` is populated only from
    ``pc_requirements.recommended``.
    """

    os: Optional[str] = None
    processor: Optional[str] = None
    memory: Optional[str] = None
    graphics: Optional[str] = None
    storage: Optional[str] = None


@dataclass(frozen=True)
class SteamFacts:
    """The subset of Steam ``appdetails`` that maps cleanly to facts.

    Every field is optional. A field that Steam did not provide is
    ``None`` — never a placeholder string and never a value copied from
    another source.
    """

    app_id: int
    name: Optional[str] = None
    release_date: Optional[str] = None
    #: The platforms Steam reports as **supported**, as canonical names.
    #:
    #: This is the list the contract models. It deliberately contains
    #: only the platforms Steam *positively* reported: a platform Steam
    #: reported as unsupported is absent from the list, exactly as it
    #: always was.
    platforms: List[str] = field(default_factory=list)
    #: The full three-state platform availability, including the
    #: platforms Steam reported as unsupported and the ones it did not
    #: report at all.
    #:
    #: The list above cannot carry that distinction on its own — an
    #: unsupported platform and an unreported one both simply do not
    #: appear in it — so the three-state view is kept alongside it.
    #: Nothing in the response contract consumes it in this stage;
    #: it exists so the extraction is honest about what Steam did and
    #: did not say.
    platform_availability: SteamPlatformAvailability = field(
        default_factory=SteamPlatformAvailability
    )
    minimum: SteamRequirements = field(default_factory=SteamRequirements)
    recommended: SteamRequirements = field(default_factory=SteamRequirements)

    def has_any_fact(self) -> bool:
        """``True`` when at least one mapped field carries a value."""

        return bool(
            self.name
            or self.release_date
            or self.platforms
            or self.minimum != SteamRequirements()
            or self.recommended != SteamRequirements()
        )


@dataclass(frozen=True)
class IdentityCheck:
    """The outcome of comparing the caller's name with Steam's title.

    ``matched`` is ``True`` only when the two titles agree under the
    documented conservative normalisation policy. When it is ``False``
    the retrieved facts describe a *different* product and must not be
    used to enrich the draft.
    """

    matched: bool
    #: The caller-supplied product name, normalised for comparison.
    requested_name: str
    #: The Steam title, normalised for comparison. ``None`` when Steam
    #: did not report a title.
    retrieved_name: Optional[str]
    #: A short, non-sensitive explanation suitable for a log line.
    reason: str


@dataclass(frozen=True)
class SteamRetrievalResult:
    """Everything the retrieval layer produced for one App ID.

    ``evidence`` is **service-owned**: it was created by the parser, not
    by a model. That distinction is what allows the service to keep
    these entries while discarding model-invented ones.
    """

    app_id: int
    facts: SteamFacts
    identity: IdentityCheck
    evidence: List[EvidenceReference] = field(default_factory=list)
    #: Evidence id -> the dotted field path it supports.
    evidence_by_field: Dict[str, str] = field(default_factory=dict)

    def evidence_id_for(self, field_path: str) -> Optional[str]:
        """Return the evidence id backing ``field_path``, if any."""

        return self.evidence_by_field.get(field_path)


__all__ = [
    "IdentityCheck",
    "SteamFacts",
    "SteamPlatformAvailability",
    "SteamRequirements",
    "SteamRetrievalResult",
]
