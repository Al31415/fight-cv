"""Loader + validators for the frozen fight-CV ontology (ontology_v1.yaml).

Import `ONTOLOGY` for the parsed dict, or the convenience frozensets for
membership checks. `version` is stamped onto every emitted row as
`ontology_version`.
"""
from __future__ import annotations

from pathlib import Path

import yaml

_ONTOLOGY_PATH = Path(__file__).resolve().parents[1] / "ontology_v1.yaml"

with _ONTOLOGY_PATH.open(encoding="utf-8") as fh:
    ONTOLOGY: dict = yaml.safe_load(fh)

VERSION: str = ONTOLOGY["version"]

ATTEMPT_FAMILIES = frozenset(ONTOLOGY["attempt_events"])
EFFECT_TYPES = frozenset(ONTOLOGY["effects"]["types"])
PHASE_STATES = frozenset(ONTOLOGY["phase"]["states"])
GROUND_POSITIONS = frozenset(ONTOLOGY["phase"]["ground_positions"])
CONTROL_STATES = frozenset(ONTOLOGY["phase"]["control"])
VISIBILITY = frozenset(ONTOLOGY["observability"]["visibility"])
SHOT_CONTENT = frozenset(ONTOLOGY["shot_state"]["content"])
REVIEW_STATUS = frozenset(ONTOLOGY["review_status"])
ABSTENTION_REASONS = frozenset(ONTOLOGY["abstention_reasons"])


def subtypes(family: str) -> frozenset[str]:
    return frozenset(ONTOLOGY["attempt_events"][family]["subtypes"])


def results(family: str) -> frozenset[str]:
    return frozenset(ONTOLOGY["attempt_events"][family]["results"])


def validate_attempt(family: str, subtype: str, result: str) -> None:
    """Raise ValueError if an attempt triple is out-of-ontology."""
    if family not in ATTEMPT_FAMILIES:
        raise ValueError(f"unknown action_family {family!r}")
    if subtype not in subtypes(family):
        raise ValueError(f"{subtype!r} not a valid {family} subtype")
    if result not in results(family):
        raise ValueError(f"{result!r} not a valid {family} result")
