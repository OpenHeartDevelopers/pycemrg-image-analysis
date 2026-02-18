# src/pycemrg_image_analysis/utilities/label_tools.py

"""
Label diagnostic and remapping tools for handling different label schemes.

Common use case: Slicer resets segmentation labels to sequential 1-N,
but library schematics expect specific label values. These tools help
users check compatibility and create custom label mappings.
"""

import logging
from pathlib import Path
from typing import Dict, List, Set, Optional
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

from pycemrg.core.logs import setup_logging
from pycemrg_image_analysis.schematics import ALL_SCHEMATICS
from pycemrg_image_analysis.utilities import load_image

setup_logging()
logger = logging.getLogger(__name__)


@dataclass
class LabelMismatch:
    """Represents a label that doesn't match between image and schematic."""
    label_name: str
    expected_value: int
    status: str  # "missing", "conflict", "ok"
    found_in_image: bool = False


@dataclass
class DiagnosticReport:
    """Results of checking image labels against schematic expectations."""
    image_path: Path
    schematic_name: str
    image_labels: Set[int]
    expected_labels: Dict[str, int]
    mismatches: List[LabelMismatch]
    
    @property
    def has_issues(self) -> bool:
        """True if any labels are missing or conflicted."""
        return any(m.status != "ok" for m in self.mismatches)
    
    @property
    def missing_labels(self) -> List[LabelMismatch]:
        """Labels expected but not found in image."""
        return [m for m in self.mismatches if m.status == "missing"]
    
    @property
    def ok_labels(self) -> List[LabelMismatch]:
        """Labels that match expectations."""
        return [m for m in self.mismatches if m.status == "ok"]


class LabelDiagnostic:
    """
    Validate image labels against schematic expectations.
    
    Helps identify when user's segmentation has different label values
    than the schematic expects (e.g., after Slicer reset to 1-N).
    """
    
    def check_image_against_schematic(
        self,
        image_path: Path,
        schematic_name: str,
    ) -> DiagnosticReport:
        """
        Compare labels in image to schematic expectations.
        
        Args:
            image_path: Path to segmentation image
            schematic_name: Name of schematic to check against
                           (e.g., "biventricular_basic", "lpv1_ring")
        
        Returns:
            DiagnosticReport with detailed mismatch information
            
        Raises:
            KeyError: If schematic_name not found
            FileNotFoundError: If image_path doesn't exist
            
        Example:
            >>> diagnostic = LabelDiagnostic()
            >>> report = diagnostic.check_image_against_schematic(
            ...     Path("seg.nrrd"),
            ...     "lpv1_ring"
            ... )
            >>> if report.has_issues:
            ...     print(f"Found {len(report.missing_labels)} missing labels")
        """
        if schematic_name not in ALL_SCHEMATICS:
            available = ", ".join(list(ALL_SCHEMATICS.keys())[:5])
            raise KeyError(
                f"Schematic '{schematic_name}' not found. "
                f"Available: {available}... (use list_available_schematics())"
            )
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load schematic expectations
        schematic = ALL_SCHEMATICS[schematic_name]
        expected_labels = schematic["labels"]
        
        # Extract unique labels from image
        img = load_image(image_path)
        img_array = sitk.GetArrayFromImage(img)
        image_labels = set(np.unique(img_array).astype(int))
        
        # Remove background (0) from comparison
        image_labels.discard(0)
        
        logger.info(f"Checking {image_path.name} against '{schematic_name}' schematic")
        logger.debug(f"  Image contains {len(image_labels)} unique labels: {sorted(image_labels)}")
        logger.debug(f"  Schematic expects {len(expected_labels)} labels")
        
        # Compare
        mismatches = []
        for label_name, expected_value in expected_labels.items():
            if expected_value in image_labels:
                status = "ok"
                found = True
            else:
                status = "missing"
                found = False
            
            mismatches.append(LabelMismatch(
                label_name=label_name,
                expected_value=expected_value,
                status=status,
                found_in_image=found
            ))
        
        return DiagnosticReport(
            image_path=image_path,
            schematic_name=schematic_name,
            image_labels=image_labels,
            expected_labels=expected_labels,
            mismatches=mismatches,
        )
    
    def print_report(self, report: DiagnosticReport) -> None:
        """
        Print a human-readable diagnostic report.
        
        Args:
            report: DiagnosticReport from check_image_against_schematic()
        """
        print(f"\n{'='*70}")
        print(f"Label Diagnostic Report")
        print(f"{'='*70}")
        print(f"Image: {report.image_path.name}")
        print(f"Schematic: {report.schematic_name}")
        print(f"\nImage contains {len(report.image_labels)} unique labels (excluding 0):")
        print(f"  {sorted(report.image_labels)}")
        print(f"\nSchematic expects {len(report.expected_labels)} labels:")
        
        if report.has_issues:
            print(f"\n❌ ISSUES FOUND:")
            print(f"\nMissing labels ({len(report.missing_labels)}):")
            for m in report.missing_labels:
                print(f"  • {m.label_name}: expected {m.expected_value}, NOT FOUND in image")
            
            if report.ok_labels:
                print(f"\n✓ Matching labels ({len(report.ok_labels)}):")
                for m in report.ok_labels:
                    print(f"  • {m.label_name}: {m.expected_value} ✓")
            
            print(f"\n💡 SUGGESTION:")
            print(f"   Your image has different label values than the schematic expects.")
            print(f"   Use LabelRemapper to create a custom mapping, or manually edit")
            print(f"   the labels.yaml file with your actual label values.")
        else:
            print(f"\n✅ ALL LABELS MATCH")
            for m in report.ok_labels:
                print(f"  • {m.label_name}: {m.expected_value} ✓")
        
        print(f"{'='*70}\n")


class LabelRemapper:
    """
    Create custom label mappings for users with different label schemes.
    
    Allows users to map their actual label values to schematic expectations
    without manually editing YAML files.
    """
    
    def create_mapping_from_dict(
        self,
        label_mapping: Dict[str, int],
    ) -> Dict[str, int]:
        """
        Validate and return a label mapping dictionary.
        
        Args:
            label_mapping: Dict mapping label names to user's integer values
                          e.g., {"LPV1_label": 1, "LA_myo_label": 5}
        
        Returns:
            Validated mapping dictionary
            
        Example:
            >>> remapper = LabelRemapper()
            >>> mapping = remapper.create_mapping_from_dict({
            ...     "LPV1_label": 1,
            ...     "LA_myo_label": 5,
            ...     "LPV1_ring_label": 50,
            ... })
        """
        # Validation
        for name, value in label_mapping.items():
            if not isinstance(value, int):
                raise ValueError(
                    f"Label value for '{name}' must be integer, got {type(value)}"
                )
            if value < 0:
                raise ValueError(
                    f"Label value for '{name}' must be non-negative, got {value}"
                )
        
        return label_mapping
    
    def suggest_mapping_from_report(
        self,
        report: DiagnosticReport,
    ) -> Optional[Dict[str, int]]:
        """
        Suggest a possible mapping based on diagnostic report.
        
        If image labels are sequential (1, 2, 3...) and match the count
        of expected labels, suggests a sequential mapping.
        
        Args:
            report: DiagnosticReport from LabelDiagnostic
        
        Returns:
            Suggested mapping dict, or None if no clear suggestion
            
        Example:
            >>> diagnostic = LabelDiagnostic()
            >>> report = diagnostic.check_image_against_schematic(...)
            >>> remapper = LabelRemapper()
            >>> suggestion = remapper.suggest_mapping_from_report(report)
            >>> if suggestion:
            ...     print("Try this mapping:", suggestion)
        """
        # Check if image labels are sequential starting from 1
        sorted_labels = sorted(report.image_labels)
        
        if len(sorted_labels) != len(report.expected_labels):
            logger.debug("Cannot suggest mapping: label count mismatch")
            return None
        
        # Check if sequential (1, 2, 3, ...)
        expected_sequence = list(range(1, len(sorted_labels) + 1))
        if sorted_labels != expected_sequence:
            logger.debug("Cannot suggest mapping: labels not sequential")
            return None
        
        # Suggest sequential mapping
        # Map expected label names to sequential values in image
        suggested = {}
        for i, (label_name, _) in enumerate(report.expected_labels.items()):
            suggested[label_name] = sorted_labels[i]
        
        logger.info("Generated sequential mapping suggestion")
        return suggested


def list_available_schematics() -> None:
    """Print all available schematic names."""
    print("\nAvailable Schematics:")
    print("=" * 50)
    
    categories = {
        "Myocardium": [k for k in ALL_SCHEMATICS.keys() 
                       if any(x in k for x in ["outflow", "wall", "artery", "myocardium", "push"])],
        "Valves": [k for k in ALL_SCHEMATICS.keys() if "valve" in k],
        "Rings": [k for k in ALL_SCHEMATICS.keys() if "ring" in k],
    }
    
    for category, names in categories.items():
        if names:
            print(f"\n{category}:")
            for name in sorted(names):
                print(f"  • {name}")
    
    print()


# Convenience function
def check_labels(image_path: Path, schematic_name: str) -> DiagnosticReport:
    """
    Quick label check - convenience wrapper.
    
    Args:
        image_path: Path to segmentation image
        schematic_name: Schematic to check against
    
    Returns:
        DiagnosticReport
        
    Example:
        >>> from pycemrg_image_analysis.utilities.label_tools import check_labels
        >>> report = check_labels(Path("seg.nrrd"), "lpv1_ring")
        >>> if report.has_issues:
        ...     print(f"{len(report.missing_labels)} labels missing")
    """
    diagnostic = LabelDiagnostic()
    report = diagnostic.check_image_against_schematic(image_path, schematic_name)
    diagnostic.print_report(report)
    return report