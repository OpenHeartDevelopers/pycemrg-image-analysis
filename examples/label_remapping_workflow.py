# examples/label_remapping_workflow.py

"""
Label Remapping Workflow Examples

Demonstrates how to handle segmentations with different label values
than library schematics expect (common after Slicer label reset).
"""

from pathlib import Path
from pycemrg_image_analysis.utilities.label_tools import (
    LabelDiagnostic,
    LabelRemapper,
    check_labels,
    list_available_schematics,
)
from pycemrg_image_analysis import ImageAnalysisScaffolder


# =============================================================================
# EXAMPLE 1: Quick Diagnostic Check
# =============================================================================

def example_quick_check():
    """
    Quickest way to check if your segmentation matches a schematic.
    """
    # One-liner diagnostic
    report = check_labels(
        image_path=Path("seg_from_slicer.nrrd"),
        schematic_name="lpv1_ring"
    )
    
    # Prints formatted report automatically
    # Shows which labels match and which are missing


# =============================================================================
# EXAMPLE 2: Detailed Diagnostic with Programmatic Access
# =============================================================================

def example_detailed_diagnostic():
    """
    Get detailed diagnostic information for custom handling.
    """
    diagnostic = LabelDiagnostic()
    
    report = diagnostic.check_image_against_schematic(
        image_path=Path("seg_from_slicer.nrrd"),
        schematic_name="lpv1_ring"
    )
    
    # Programmatic access to results
    if report.has_issues:
        print(f"Found {len(report.missing_labels)} missing labels")
        
        for mismatch in report.missing_labels:
            print(f"  Expected {mismatch.label_name}={mismatch.expected_value}")
        
        print(f"\nYour image has: {sorted(report.image_labels)}")
        
        # Suggest next steps
        remapper = LabelRemapper()
        suggestion = remapper.suggest_mapping_from_report(report)
        
        if suggestion:
            print("\n💡 Auto-detected sequential labels!")
            print("   Try this mapping:")
            for name, value in suggestion.items():
                print(f"     {name}: {value}")
    else:
        print("✅ All labels match - ready to use schematic as-is")


# =============================================================================
# EXAMPLE 3: Manual Mapping Creation
# =============================================================================

def example_manual_mapping():
    """
    Create custom label mapping when Slicer resets to 1-N.
    """
    # Your segmentation after Slicer reset
    # Slicer changed: LPV1=8 → 1, LA_myo=104 → 5, etc.
    
    # Define your actual label values
    my_label_mapping = {
        "LPV1_label": 1,        # Slicer reset this to 1 (was 8)
        "LA_myo_label": 5,      # Slicer reset this to 5 (was 104)
        "LPV1_ring_label": 50,  # Choose output label for ring
    }
    
    # Scaffold with custom mapping
    scaffolder = ImageAnalysisScaffolder()
    scaffolder.scaffold_components_with_mapping(
        output_dir=Path("config/"),
        component_names=["lpv1_ring"],
        label_mapping=my_label_mapping,
    )
    
    print("✅ Config generated with your custom labels")
    print("   labels.yaml now contains: 1, 5, 50 instead of 8, 104, 208")


# =============================================================================
# EXAMPLE 4: Check First, Then Map
# =============================================================================

def example_check_and_map_workflow():
    """
    Complete workflow: diagnose, then create mapping if needed.
    """
    image_path = Path("seg_from_slicer.nrrd")
    schematic_name = "biventricular_basic"
    
    # Step 1: Check what we're working with
    print("Step 1: Diagnostic check...")
    diagnostic = LabelDiagnostic()
    report = diagnostic.check_image_against_schematic(image_path, schematic_name)
    diagnostic.print_report(report)
    
    # Step 2: Decide on mapping strategy
    if not report.has_issues:
        print("\n✅ Labels match - using standard scaffolding")
        scaffolder = ImageAnalysisScaffolder()
        scaffolder.scaffold_components(
            output_dir=Path("config/"),
            component_names=["biventricular_basic"]
        )
        return
    
    # Step 3: Try auto-suggestion
    print("\nStep 2: Checking for auto-mapping...")
    remapper = LabelRemapper()
    suggestion = remapper.suggest_mapping_from_report(report)
    
    if suggestion:
        print("✅ Auto-detected sequential mapping!")
        print("   Using suggested mapping:")
        for name, value in list(suggestion.items())[:5]:
            print(f"     {name}: {value}")
        if len(suggestion) > 5:
            print(f"     ... and {len(suggestion) - 5} more")
        
        # Use suggested mapping
        scaffolder = ImageAnalysisScaffolder()
        scaffolder.scaffold_components_with_mapping(
            output_dir=Path("config/"),
            component_names=["biventricular_basic"],
            label_mapping=suggestion,
        )
    else:
        print("⚠️  Cannot auto-detect mapping")
        print("   You'll need to create a manual mapping:")
        print("\n   Example:")
        print("   my_mapping = {")
        for label_name in list(report.expected_labels.keys())[:3]:
            print(f'       "{label_name}": ???,  # Your value here')
        print("       ...")
        print("   }")
        print("\n   Then use scaffolder.scaffold_components_with_mapping(...)")


# =============================================================================
# EXAMPLE 5: Batch Check Multiple Schematics
# =============================================================================

def example_batch_diagnostic():
    """
    Check one image against multiple schematics.
    """
    image_path = Path("my_seg.nrrd")
    schematics_to_check = ["lpv1_ring", "lpv2_ring", "mitral_valve"]
    
    diagnostic = LabelDiagnostic()
    
    results = {}
    for schematic_name in schematics_to_check:
        report = diagnostic.check_image_against_schematic(
            image_path, schematic_name
        )
        results[schematic_name] = report.has_issues
    
    # Summary
    print("\n" + "="*50)
    print("Batch Diagnostic Summary")
    print("="*50)
    for name, has_issues in results.items():
        status = "❌ Issues" if has_issues else "✅ OK"
        print(f"{name:20} {status}")


# =============================================================================
# EXAMPLE 6: List Available Schematics
# =============================================================================

def example_list_schematics():
    """
    Show all available schematics before checking.
    """
    from pycemrg_image_analysis.utilities.label_tools import list_available_schematics
    
    list_available_schematics()


if __name__ == "__main__":
    # Run examples
    print("Example 1: Quick Check")
    print("-" * 50)
    # example_quick_check()
    
    print("\n\nExample 2: Detailed Diagnostic")
    print("-" * 50)
    # example_detailed_diagnostic()
    
    print("\n\nExample 3: Manual Mapping")
    print("-" * 50)
    # example_manual_mapping()
    
    print("\n\nExample 4: Full Workflow")
    print("-" * 50)
    # example_check_and_map_workflow()
    
    print("\n\nExample 6: List Schematics")
    print("-" * 50)
    example_list_schematics()