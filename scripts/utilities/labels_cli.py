#!/usr/bin/env python
# scripts/utilities/labels.py

"""
Command-line tool for label diagnostics and remapping.

Usage:
    python scripts/utilities/labels.py show --input IMAGE
    python scripts/utilities/labels.py check --input IMAGE --schematic NAME
    python scripts/utilities/labels.py list-schematics
    python scripts/utilities/labels.py suggest --input IMAGE --schematic NAME
    python scripts/utilities/labels.py volumes --input IMAGE --labels 1 2 3
    python scripts/utilities/labels.py volumes --input IMAGE --labels-file labels.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pycemrg.data import LabelManager

from pycemrg_image_analysis.utilities.label_tools import (
    LabelDiagnostic,
    LabelRemapper,
    list_available_schematics,
)
from pycemrg_image_analysis.utilities import load_image
from pycemrg_image_analysis.utilities.postprocessing import compute_label_volumes


def cmd_show(args):
    """Show all unique labels in an image."""
    print(f"\nAnalyzing: {args.input}")
    print("=" * 70)
    
    # Load image
    try:
        img = load_image(args.input)
    except FileNotFoundError:
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
    
    # Extract labels
    img_array = sitk.GetArrayFromImage(img)
    unique_labels = np.unique(img_array).astype(int)
    
    # Count voxels per label
    label_counts = {}
    for label in unique_labels:
        count = np.sum(img_array == label)
        label_counts[label] = count
    
    # Display
    print(f"\nImage shape (Z, Y, X): {img_array.shape}")
    print(f"Spacing (X, Y, Z): {img.GetSpacing()}")
    print(f"Origin (X, Y, Z): {img.GetOrigin()}")
    
    print(f"\nFound {len(unique_labels)} unique labels:")
    print(f"{'Label':<10} {'Voxel Count':<15} {'Percentage':<10}")
    print("-" * 40)
    
    total_voxels = img_array.size
    for label in sorted(unique_labels):
        count = label_counts[label]
        percentage = (count / total_voxels) * 100
        
        # Highlight non-zero labels
        if label == 0:
            marker = ""
        else:
            marker = "  ●"
        
        print(f"{label:<10} {count:<15} {percentage:>6.2f}%{marker}")
    
    # Summary
    non_zero_labels = [l for l in unique_labels if l != 0]
    if non_zero_labels:
        print(f"\n✓ Non-zero labels: {sorted(non_zero_labels)}")
    else:
        print(f"\nWarning: No non-zero labels found (image may be blank)")
    
    print()


def cmd_check(args):
    """Check image labels against schematic expectations."""
    print(f"\nChecking: {args.input}")
    print(f"Against schematic: {args.schematic}")
    print("=" * 70)
    
    try:
        diagnostic = LabelDiagnostic()
        report = diagnostic.check_image_against_schematic(
            Path(args.input),
            args.schematic
        )
        diagnostic.print_report(report)
        
        # Exit code: 0 if OK, 1 if issues
        sys.exit(1 if report.has_issues else 0)
        
    except FileNotFoundError:
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: {e}")
        print("\nRun 'python scripts/utilities/labels.py list-schematics' to see available schematics")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_list_schematics(args):
    """List all available schematics."""
    list_available_schematics()


def cmd_suggest(args):
    """Suggest label mapping based on diagnostic."""
    print(f"\nAnalyzing: {args.input}")
    print(f"For schematic: {args.schematic}")
    print("=" * 70)
    
    try:
        # Run diagnostic
        diagnostic = LabelDiagnostic()
        report = diagnostic.check_image_against_schematic(
            Path(args.input),
            args.schematic
        )
        
        if not report.has_issues:
            print("\nLabels already match schematic - no mapping needed")
            sys.exit(0)
        
        # Try to suggest mapping
        remapper = LabelRemapper()
        suggestion = remapper.suggest_mapping_from_report(report)
        
        if suggestion:
            print("\nAuto-detected sequential mapping:")
            print("\nCopy this mapping into your code:")
            print("-" * 70)
            print("mapping = {")
            for name, value in suggestion.items():
                print(f'    "{name}": {value},')
            print("}")
            print("-" * 70)
            
            print("\nOr save to file:")
            if args.output:
                import json
                output_path = Path(args.output)
                with open(output_path, 'w') as f:
                    json.dump(suggestion, f, indent=2)
                print(f"✓ Saved mapping to: {output_path}")
            else:
                print("  Use --output FILENAME.json to save mapping")
            
            sys.exit(0)
        else:
            print("\nCannot auto-detect mapping")
            print("\nReasons:")
            print("  - Labels are not sequential (1, 2, 3...)")
            print("  - Label count doesn't match schematic expectations")
            
            print("\nYour image has these labels:")
            print(f"  {sorted(report.image_labels)}")
            
            print("\nSchematic expects these labels:")
            for name, value in report.expected_labels.items():
                print(f"  {name}: {value}")
            
            print("\nYou'll need to create a manual mapping:")
            print("-" * 70)
            print("mapping = {")
            for name in report.expected_labels.keys():
                print(f'    "{name}": ???,  # Replace ??? with your label value')
            print("}")
            print("-" * 70)
            
            sys.exit(1)
            
    except FileNotFoundError:
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_volumes(args):
    """Compute physical volumes for a set of labels."""
    try:
        img = load_image(args.input)
    except FileNotFoundError:
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    # Resolve label values and build name lookup
    name_lookup: dict[int, str] = {}

    if args.labels_file:
        try:
            with open(args.labels_file) as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: Labels file not found: {args.labels_file}")
            sys.exit(1)

        raw = config.get("labels", {})
        if not raw:
            print(f"Error: No 'labels' key found in {args.labels_file}")
            sys.exit(1)

        label_values = [int(v) for v in raw.values()]
        label_manager = LabelManager(config_path=Path(args.labels_file))
        for value in label_values:
            try:
                name_lookup[value] = label_manager.get_name(value)
            except KeyError:
                name_lookup[value] = f"label_{value}"
    else:
        label_values = args.labels

    try:
        volumes = compute_label_volumes(img, label_values, use_mm3=args.mm3)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.plain:
        for value, vol in volumes.by_label.items():
            name = name_lookup.get(value, f"label_{value}")
            print(f"{value},{name},{vol:.4f},{volumes.unit}")
        print(f"total,total,{volumes.total:.4f},{volumes.unit}")
        return

    result = {
        "input": str(args.input),
        "unit": volumes.unit,
        "labels": [
            {
                "value": value,
                "name": name_lookup.get(value, f"label_{value}"),
                "volume": vol,
            }
            for value, vol in volumes.by_label.items()
        ],
        "total": volumes.total,
    }
    json_str = json.dumps(result, indent=2)

    if args.output:
        try:
            with open(args.output, "w") as f:
                f.write(json_str)
        except OSError as e:
            print(f"Error writing output: {e}")
            sys.exit(1)
    else:
        print(json_str)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Label diagnostic and remapping utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show labels in an image
  python scripts/utilities/labels.py show --input seg.nrrd
  
  # Check image against schematic
  python scripts/utilities/labels.py check --input seg.nrrd --schematic lpv1_ring
  
  # List available schematics
  python scripts/utilities/labels.py list-schematics
  
  # Auto-suggest mapping
  python scripts/utilities/labels.py suggest --input seg.nrrd --schematic lpv1_ring
  
  # Auto-suggest and save to file
  python scripts/utilities/labels.py suggest --input seg.nrrd --schematic lpv1_ring --output mapping.json

  # Compute volumes by integer label (default mL)
  python scripts/utilities/labels.py volumes --input seg.nrrd --labels 1 2 3

  # Compute volumes from labels.yaml, report in mm3
  python scripts/utilities/labels.py volumes --input seg.nrrd --labels-file labels.yaml --mm3

  # Plain output for CSV accumulation
  python scripts/utilities/labels.py volumes --input seg.nrrd --labels 1 2 3 --plain >> results.csv

  # Write JSON to file
  python scripts/utilities/labels.py volumes --input seg.nrrd --labels-file labels.yaml --output volumes.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    subparsers.required = True
    
    # --- SHOW command ---
    show_parser = subparsers.add_parser(
        'show',
        help='Show all unique labels in an image'
    )
    show_parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Path to image file (.nrrd, .nii, .nii.gz)'
    )
    show_parser.set_defaults(func=cmd_show)
    
    # --- CHECK command ---
    check_parser = subparsers.add_parser(
        'check',
        help='Check image labels against schematic expectations'
    )
    check_parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Path to image file'
    )
    check_parser.add_argument(
        '--schematic',
        type=str,
        required=True,
        help='Schematic name (e.g., lpv1_ring, biventricular_basic)'
    )
    check_parser.set_defaults(func=cmd_check)
    
    # --- LIST-SCHEMATICS command ---
    list_parser = subparsers.add_parser(
        'list-schematics',
        help='List all available schematics'
    )
    list_parser.set_defaults(func=cmd_list_schematics)
    
    # --- SUGGEST command ---
    suggest_parser = subparsers.add_parser(
        'suggest',
        help='Auto-suggest label mapping (for sequential labels)'
    )
    suggest_parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Path to image file'
    )
    suggest_parser.add_argument(
        '--schematic',
        type=str,
        required=True,
        help='Schematic name'
    )
    suggest_parser.add_argument(
        '--output',
        type=Path,
        required=False,
        help='Save mapping to JSON file (optional)'
    )
    suggest_parser.set_defaults(func=cmd_suggest)

    # --- VOLUMES command ---
    volumes_parser = subparsers.add_parser(
        'volumes',
        help='Compute physical volume of one or more labels'
    )
    volumes_parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Path to image file (.nrrd, .nii, .nii.gz)'
    )
    label_source = volumes_parser.add_mutually_exclusive_group(required=True)
    label_source.add_argument(
        '--labels',
        type=int,
        nargs='+',
        metavar='LABEL',
        help='Integer label values to measure (e.g. --labels 1 2 3)'
    )
    label_source.add_argument(
        '--labels-file',
        type=Path,
        metavar='FILE',
        help='Path to labels.yaml (LabelManager format); measures all labels in file'
    )
    volumes_parser.add_argument(
        '--mm3',
        action='store_true',
        help='Report volumes in mm³ instead of mL'
    )
    volumes_parser.add_argument(
        '--plain',
        action='store_true',
        help='Plain CSV-style output (value,name,volume,unit) instead of JSON'
    )
    volumes_parser.add_argument(
        '--output',
        type=Path,
        required=False,
        help='Write JSON output to file (cannot be combined with --plain)'
    )
    volumes_parser.set_defaults(func=cmd_volumes)

    # Parse and execute
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()