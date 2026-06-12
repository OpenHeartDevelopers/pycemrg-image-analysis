# src/pycemrg_image_analysis/utilities/schematic_tools.py

"""
Stateless helpers for *inspecting* the schematic catalogue and *reconciling* a
schematic's label vocabulary against an existing ``labels.yaml``.

These functions take plain data (the schematic dicts, an existing name->int map)
and return data or human-readable strings. They never touch the filesystem or
read process state — orchestration (argv parsing, file I/O, printing) lives in
``pycemrg_image_analysis.cli``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import yaml

from pycemrg_image_analysis.schematics.myocardium import MYOCARDIUM_SCHEMATICS
from pycemrg_image_analysis.schematics.rings import RING_SCHEMATICS
from pycemrg_image_analysis.schematics.valves import VALVE_SCHEMATICS
from pycemrg_image_analysis.logic.constants import (
    MyocardiumSemanticRole,
    RingSemanticRole,
    ValveSemanticRole,
)

# Authoritative category -> {component_name: schematic} mapping. Driven by the
# source-of-truth family dicts rather than substring guessing on names.
CATEGORIES: Dict[str, Dict[str, dict]] = {
    "myocardium": MYOCARDIUM_SCHEMATICS,
    "valves": VALVE_SCHEMATICS,
    "rings": RING_SCHEMATICS,
}

# Role-enum members that carry the ordered list of mask application steps.
_APPLICATION_STEP_ROLES = {
    MyocardiumSemanticRole.APPLICATION_STEPS,
    ValveSemanticRole.APPLICATION_STEPS,
    RingSemanticRole.APPLICATION_STEPS,
}


# ---------------------------------------------------------------------------
# Family / target resolution
# ---------------------------------------------------------------------------
def category_of(component_name: str) -> str:
    """Return the category a component belongs to, or '' if unknown."""
    for category, schematics in CATEGORIES.items():
        if component_name in schematics:
            return category
    return ""


def resolve_target(token: str) -> Tuple[str, List[str], Dict[str, dict]]:
    """
    Resolve a CLI token to a set of schematics.

    The token may be a *category* name (``myocardium`` / ``valves`` / ``rings``)
    or a single *component* name (e.g. ``lv_outflow``).

    Returns:
        (category, component_names, {name: schematic}) — ``category`` is the
        family name; for a single component it is that component's family.

    Raises:
        KeyError: if the token matches neither a category nor a component.
    """
    if token in CATEGORIES:
        schematics = CATEGORIES[token]
        return token, list(schematics.keys()), dict(schematics)

    category = category_of(token)
    if category:
        schematic = CATEGORIES[category][token]
        return category, [token], {token: schematic}

    known = ", ".join(sorted(CATEGORIES)) + ", or a component name"
    raise KeyError(f"Unknown schematic target '{token}'. Expected one of: {known}")


# ---------------------------------------------------------------------------
# Readable summaries
# ---------------------------------------------------------------------------
def _humanize_role(role: Enum) -> str:
    """'SOURCE_BLOOD_POOL_NAME' -> 'Source blood pool'."""
    text = role.name.lower().replace("_", " ")
    for suffix in (" parameter name", " name"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return text[:1].upper() + text[1:]


def render_application_step(step: dict) -> str:
    """Render one ``{'MODE': ..., 'RULE_LABEL_NAMES': [...]}`` step in English."""
    mode = step.get("MODE", "?")
    names = step.get("RULE_LABEL_NAMES", []) or []
    joined = ", ".join(names) if names else "(none)"
    if mode == "ADD":
        return "ADD — write the new label everywhere the grown mask is set"
    if mode == "REPLACE":
        return "REPLACE — overwrite the intersection region with the new label"
    if mode == "REPLACE_ONLY":
        return f"REPLACE_ONLY — write only over: {joined}"
    if mode == "REPLACE_EXCEPT":
        return f"REPLACE_EXCEPT — write everywhere in the mask except over: {joined}"
    return f"{mode} — labels: {joined}"


def summarize_schematic(name: str, schematic: dict) -> str:
    """Return a multi-line, human-readable summary of a single schematic."""
    labels: Dict[str, int] = schematic.get("labels", {})
    parameters: Dict[str, float] = schematic.get("parameters", {})
    semantic_map: dict = schematic.get("semantic_map", {})

    lines: List[str] = [name, "-" * len(name)]

    if not semantic_map:
        lines.append(
            "  (parameter/label-only schematic — no semantic map; supplies "
            "labels and thicknesses for push steps)"
        )
    else:
        lines.append("  Roles:")
        for role, target in semantic_map.items():
            if role in _APPLICATION_STEP_ROLES:
                continue
            lines.append(f"    {_humanize_role(role)}: {target}")

    if labels:
        lines.append("  Labels (name → voxel value):")
        for label_name, value in labels.items():
            lines.append(f"    {label_name}: {value}")

    if parameters:
        lines.append("  Parameters:")
        for param_name, value in parameters.items():
            lines.append(f"    {param_name}: {value}")

    steps = next(
        (v for role, v in semantic_map.items() if role in _APPLICATION_STEP_ROLES),
        None,
    )
    if steps:
        lines.append("  Application steps (applied in order):")
        for index, step in enumerate(steps, start=1):
            lines.append(f"    {index}. {render_application_step(step)}")
    elif steps == []:
        lines.append("  Application steps: none")

    return "\n".join(lines)


def render_category_overview() -> str:
    """Return a short listing of every category and its component names."""
    lines = ["Available schematic categories:", "=" * 40]
    for category, schematics in CATEGORIES.items():
        lines.append(f"\n{category} ({len(schematics)}):")
        for name in sorted(schematics):
            lines.append(f"  • {name}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Label reconciliation against an existing labels.yaml
# ---------------------------------------------------------------------------
class LabelStatus(Enum):
    """How a schematic label relates to the user's existing labels.yaml."""

    REUSED = "reused"            # name present with the same integer value
    VALUE_MISMATCH = "mismatch"  # name present but a different integer value
    NEW = "new"                  # name absent — a fresh entry is proposed


@dataclass(frozen=True)
class ReconciledLabel:
    """Outcome of reconciling one schematic label name against labels.yaml."""

    name: str
    value: int                       # the value to use (existing wins; else assigned)
    status: LabelStatus
    schematic_default: int           # the value the schematic ships with
    possible_conflicts: Tuple[str, ...] = ()  # existing names already at the wanted value


def reconcile_labels(
    schematic_labels: Dict[str, int],
    existing_labels: Dict[str, int],
) -> List[ReconciledLabel]:
    """
    Classify each schematic label against an existing ``{name: int}`` map.

    New names are assigned the schematic default value if it is free, otherwise
    the next free integer. When a new name's wanted (default) value is *already
    used* by a differently-named existing label, those names are recorded as
    ``possible_conflicts`` — a strong hint that the same structure already
    exists under another name (e.g. the schematic wants ``LV_BP_label: 1`` but
    you already have ``LV_bloodpool: 1``). Full-string name similarity is a poor
    signal here because cardiac names share ``_label``/``BP`` boilerplate, so we
    key off the integer value instead, which the printed merged file lets you
    confirm visually.
    """
    taken = set(existing_labels.values())
    value_to_names: Dict[int, List[str]] = {}
    for label_name, label_value in existing_labels.items():
        value_to_names.setdefault(label_value, []).append(label_name)

    def next_free(preferred: int) -> int:
        if preferred not in taken:
            return preferred
        candidate = (max(taken) + 1) if taken else 1
        while candidate in taken:
            candidate += 1
        return candidate

    results: List[ReconciledLabel] = []
    for name, default in schematic_labels.items():
        if name in existing_labels:
            existing_value = existing_labels[name]
            status = (
                LabelStatus.REUSED
                if existing_value == default
                else LabelStatus.VALUE_MISMATCH
            )
            results.append(ReconciledLabel(name, existing_value, status, default))
        else:
            value = next_free(default)
            taken.add(value)
            conflicts = tuple(value_to_names.get(default, []))
            results.append(
                ReconciledLabel(name, value, LabelStatus.NEW, default, conflicts)
            )
    return results


def merged_labels(
    existing_labels: Dict[str, int],
    reconciled: List[ReconciledLabel],
) -> Dict[str, int]:
    """Existing labels preserved, with reconciled values layered on top."""
    merged = dict(existing_labels)
    for item in reconciled:
        merged[item.name] = item.value
    return merged


def merged_labels_yaml(
    existing_labels: Dict[str, int],
    reconciled: List[ReconciledLabel],
) -> str:
    """Render the merged labels as a ``labels.yaml`` body."""
    body = {"labels": merged_labels(existing_labels, reconciled)}
    return yaml.dump(body, sort_keys=False)


def render_reconcile_report(reconciled: List[ReconciledLabel]) -> str:
    """Group the reconciliation outcome into a readable review report."""
    reused = [r for r in reconciled if r.status is LabelStatus.REUSED]
    mismatched = [r for r in reconciled if r.status is LabelStatus.VALUE_MISMATCH]
    new = [r for r in reconciled if r.status is LabelStatus.NEW]
    conflicts = [r for r in new if r.possible_conflicts]

    lines = ["Label reconciliation:", "=" * 40]

    if reused:
        lines.append(f"\nReused (already in labels.yaml, value matches) — {len(reused)}:")
        for r in reused:
            lines.append(f"  ✓ {r.name}: {r.value}")

    if mismatched:
        lines.append(f"\nValue mismatch (keeping your value) — {len(mismatched)}:")
        for r in mismatched:
            lines.append(
                f"  ! {r.name}: {r.value}  (schematic default is {r.schematic_default})"
            )

    if new:
        lines.append(f"\nNew (proposed entries) — {len(new)}:")
        for r in new:
            note = "" if r.value == r.schematic_default else "  (default was taken)"
            lines.append(f"  + {r.name}: {r.value}{note}")

    if conflicts:
        lines.append("\n⚠ Possible duplicate structures (value already in use):")
        for r in conflicts:
            near = ", ".join(r.possible_conflicts)
            lines.append(
                f"  ⚠ new '{r.name}' wants value {r.schematic_default}, "
                f"already held by: {near} (reassigned to {r.value})"
            )
        lines.append(
            "  → if these name the same structure, rename in your labels.yaml or "
            "the schematic before scaffolding."
        )

    return "\n".join(lines)
