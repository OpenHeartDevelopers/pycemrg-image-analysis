# src/pycemrg_image_analysis/cli.py

"""
``pycemrg-ima`` — a quality-of-life CLI for understanding and templating the
schematic catalogue (myocardium, valves, rings).

    pycemrg-ima inspect                  # list every category and its components
    pycemrg-ima inspect myocardium       # summarise a whole family
    pycemrg-ima inspect lv_outflow       # summarise a single component
    pycemrg-ima create myocardium        # print a labels template for review
    pycemrg-ima create lv_outflow -o cfg # also write/merge config into cfg/

`create` reconciles the schematic's label vocabulary against an existing
`labels.yaml` (the source of truth for integer values), prints the result for
review — flagging misspellings / conflicting names — and only writes files when
`-o` is given.

This module is orchestration: argv parsing, file I/O and printing. The data
transforms live in ``utilities.schematic_tools``; config writing reuses
``ImageAnalysisScaffolder``.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import yaml

from pycemrg_image_analysis.utilities.schematic_tools import (
    render_category_overview,
    render_reconcile_report,
    reconcile_labels,
    resolve_target,
    summarize_schematic,
    merged_labels,
)


def _load_existing_labels(path: Path) -> Dict[str, int]:
    """Read the ``labels`` block from a labels.yaml; {} if absent/empty."""
    if not path.exists():
        return {}
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    return {name: int(value) for name, value in (config.get("labels") or {}).items()}


def _load_existing_parameters(path: Path) -> Dict[str, float]:
    """Read an existing parameters.json; {} if absent."""
    if not path.exists():
        return {}
    with open(path) as handle:
        return json.load(handle) or {}


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------
def cmd_inspect(args: argparse.Namespace) -> None:
    if not args.target:
        print(render_category_overview())
        return

    try:
        category, component_names, schematics = resolve_target(args.target)
    except KeyError as error:
        print(f"Error: {error}")
        sys.exit(1)

    print(f"\nCategory: {category}")
    print("=" * 40)
    for name in component_names:
        print()
        print(summarize_schematic(name, schematics[name]))
    print()


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------
def cmd_create(args: argparse.Namespace) -> None:
    try:
        category, component_names, schematics = resolve_target(args.target)
    except KeyError as error:
        print(f"Error: {error}")
        sys.exit(1)

    # Union of labels/parameters across the resolved components.
    union_labels: Dict[str, int] = {}
    union_parameters: Dict[str, float] = {}
    for name in component_names:
        union_labels.update(schematics[name]["labels"])
        union_parameters.update(schematics[name]["parameters"])

    # Decide which existing labels.yaml to reconcile against.
    output_dir = Path(args.output) if args.output else None
    if args.labels:
        labels_path = Path(args.labels)
    elif output_dir is not None:
        labels_path = output_dir / "labels.yaml"
    else:
        labels_path = None

    existing_labels = _load_existing_labels(labels_path) if labels_path else {}

    reconciled = reconcile_labels(union_labels, existing_labels)

    # --- Always print for review (the headline behaviour) ---
    print(f"\nTemplate for: {category} — components: {', '.join(component_names)}")
    if labels_path:
        where = "found" if labels_path.exists() else "not found, starting empty"
        print(f"Existing labels.yaml: {labels_path} ({where})")
    else:
        print("Existing labels.yaml: none provided (--labels) — showing schematic defaults")
    print("=" * 70)
    print()
    print(render_reconcile_report(reconciled))
    print()
    print("Resulting labels.yaml:")
    print("-" * 40)
    print(yaml.dump({"labels": merged_labels(existing_labels, reconciled)}, sort_keys=False), end="")
    print("-" * 40)
    print("Parameters:")
    print(json.dumps(union_parameters, indent=2))

    if output_dir is None:
        print("\n(Nothing written. Re-run with -o OUTPUT_DIR to scaffold config files.)")
        return

    _write_config(output_dir, component_names, existing_labels, reconciled, union_parameters)


def _write_config(
    output_dir: Path,
    component_names: list[str],
    existing_labels: Dict[str, int],
    reconciled,
    union_parameters: Dict[str, float],
) -> None:
    """Scaffold config into ``output_dir``, merging into any existing files."""
    from pycemrg_image_analysis.scaffolding import ImageAnalysisScaffolder

    existing_parameters = _load_existing_parameters(output_dir / "parameters.json")
    mapping = {item.name: item.value for item in reconciled}

    # Reuse the scaffolder to emit semantic_maps + a first pass of the config.
    scaffolder = ImageAnalysisScaffolder()
    scaffolder.scaffold_components_with_mapping(
        output_dir=output_dir,
        component_names=component_names,
        label_mapping=mapping,
        overwrite=True,
    )

    # Rewrite labels.yaml / parameters.json so pre-existing, unrelated entries
    # survive (the scaffolder rebuilds these from the schematic alone).
    full_labels = merged_labels(existing_labels, reconciled)
    full_parameters = {**existing_parameters, **union_parameters}
    with open(output_dir / "labels.yaml", "w") as handle:
        yaml.dump({"labels": full_labels}, handle, sort_keys=False)
    with open(output_dir / "parameters.json", "w") as handle:
        json.dump(full_parameters, handle, indent=2)

    print(f"\nMerged labels.yaml and parameters.json preserved in '{output_dir.resolve()}'.")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pycemrg-ima",
        description="Inspect and template the schematic catalogue.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pycemrg-ima inspect                     # list categories and components
  pycemrg-ima inspect myocardium          # summarise a whole family
  pycemrg-ima inspect lv_outflow          # summarise a single component
  pycemrg-ima create rings                # print a labels template for review
  pycemrg-ima create la_myocardium --labels config/labels.yaml
  pycemrg-ima create lv_outflow -o config # write/merge config into config/
        """,
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    inspect_parser = subparsers.add_parser(
        "inspect", help="Show schematics in a readable form"
    )
    inspect_parser.add_argument(
        "target",
        nargs="?",
        help="Category (myocardium/valves/rings) or component name. "
        "Omit to list everything.",
    )
    inspect_parser.set_defaults(func=cmd_inspect)

    create_parser = subparsers.add_parser(
        "create",
        help="Print (and optionally write) a labels template reconciled "
        "against an existing labels.yaml",
    )
    create_parser.add_argument(
        "target", help="Category (myocardium/valves/rings) or component name"
    )
    create_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Config directory to write/merge into. If omitted, the template "
        "is only printed.",
    )
    create_parser.add_argument(
        "--labels",
        type=Path,
        help="Existing labels.yaml to reconcile against (defaults to "
        "<output>/labels.yaml when -o is given).",
    )
    create_parser.set_defaults(func=cmd_create)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
