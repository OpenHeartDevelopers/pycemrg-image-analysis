# src/pycemrg_image_analysis/scaffolding.py

import json
import yaml
from pathlib import Path
from typing import Union

# Import the base class we will be extending
from pycemrg.files import ConfigScaffolder
from pycemrg_image_analysis.logic.constants import MyocardiumSemanticRole

# schematics
_LV_OUTFLOW_SCHEMATIC = {
    "labels": {"LV_BP_label": 1, "LV_myo_label": 2},
    "parameters": {"LV_neck_WT": 2.0},
    "semantic_map": {
        MyocardiumSemanticRole.SOURCE_BLOOD_POOL_NAME: "LV_BP_label",
        MyocardiumSemanticRole.TARGET_MYOCARDIUM_NAME: "LV_myo_label",
        MyocardiumSemanticRole.WALL_THICKNESS_PARAMETER_NAME: "LV_neck_WT",
        MyocardiumSemanticRole.APPLICATION_STEPS: [
            {
                "MODE": "ADD", 
                "RULE_LABEL_NAMES": []
             }
        ],
    },
}

_AORTIC_WALL_SCHEMATIC = {
    "labels": {
        "LV_BP_label": 1,
        "LV_myo_label": 2,
        "Ao_BP_label": 6,
        "Ao_wall_label": 106,
    },
    "parameters": {
        "Ao_WT": 2.0,
    },
    "semantic_map": {
        MyocardiumSemanticRole.SOURCE_BLOOD_POOL_NAME: "Ao_BP_label",
        MyocardiumSemanticRole.TARGET_MYOCARDIUM_NAME: "Ao_wall_label",
        MyocardiumSemanticRole.WALL_THICKNESS_PARAMETER_NAME: "Ao_WT",
        
        # This key is now APPLICATION_STEPS and its value is a list of step objects
        MyocardiumSemanticRole.APPLICATION_STEPS: [
            {
                "MODE": "REPLACE_EXCEPT",
                "RULE_LABEL_NAMES": [
                    "LV_BP_label",
                    "LV_myo_label",
                    "Ao_BP_label",
                ],
            }
        ],
    },
}

_PULMONARY_ARTERY_SCHEMATIC = {
    "labels": {
        "PArt_BP_label": 7, "PArt_wall_label": 107, "RV_BP_label": 3,
        "Ao_wall_label": 106, "Ao_BP_label": 6,
    },
    "parameters": {"PArt_WT": 2.0},
    "semantic_map": {
        MyocardiumSemanticRole.SOURCE_BLOOD_POOL_NAME: "PArt_BP_label",
        MyocardiumSemanticRole.TARGET_MYOCARDIUM_NAME: "PArt_wall_label",
        MyocardiumSemanticRole.WALL_THICKNESS_PARAMETER_NAME: "PArt_WT",
        MyocardiumSemanticRole.APPLICATION_STEPS: [
            {
                "MODE": "REPLACE_EXCEPT", 
                "RULE_LABEL_NAMES": ["RV_BP_label", "Ao_wall_label", "Ao_BP_label"]
            }
        ],
    },
}

_RV_MYOCARDIUM_SCHEMATIC = {
    "labels": {
        "RV_BP_label": 3,
        "RV_myo_label": 103,
        "Ao_wall_label": 106,        
    }, 
    "parameters": {
        "RV_WT": 3.5,
    },
    "semantic_map": {
        MyocardiumSemanticRole.SOURCE_BLOOD_POOL_NAME: "RV_BP_label",
        MyocardiumSemanticRole.TARGET_MYOCARDIUM_NAME: "RV_myo_label",
        MyocardiumSemanticRole.WALL_THICKNESS_PARAMETER_NAME: "RV_WT",

        # This key is now APPLICATION_STEPS and its value is a list of step objects
        MyocardiumSemanticRole.APPLICATION_STEPS: [
            {
                "MODE": "REPLACE_ONLY",
                "RULE_LABEL_NAMES": [
                    "Ao_wall_label",
                ],
            }
        ],
    }
}

_LA_MYOCARDIUM_SCHEMATIC = {
    "labels": {"LA_BP_label": 4, "LA_myo_label": 104, "RA_BP_label": 5, "SVC_label": 13},
    "parameters": {"LA_WT": 2.0},
    "semantic_map": {
        MyocardiumSemanticRole.SOURCE_BLOOD_POOL_NAME: "LA_BP_label",
        MyocardiumSemanticRole.TARGET_MYOCARDIUM_NAME: "LA_myo_label",
        MyocardiumSemanticRole.WALL_THICKNESS_PARAMETER_NAME: "LA_WT",
        MyocardiumSemanticRole.APPLICATION_STEPS: [
            {
                "MODE": "REPLACE_ONLY", 
                "RULE_LABEL_NAMES": ["RA_BP_label", "SVC_label"]
            }
        ],
    },
}

_RA_MYOCARDIUM_SCHEMATIC = {
    "labels": {"RA_BP_label": 5, "RA_myo_label": 105, "RPV1_label": 10},
    "parameters": {"RA_WT": 2.0},
    "semantic_map": {
        MyocardiumSemanticRole.SOURCE_BLOOD_POOL_NAME: "RA_BP_label",
        MyocardiumSemanticRole.TARGET_MYOCARDIUM_NAME: "RA_myo_label",
        MyocardiumSemanticRole.WALL_THICKNESS_PARAMETER_NAME: "RA_WT",
        MyocardiumSemanticRole.APPLICATION_STEPS: [
            {
                "MODE": "REPLACE_ONLY", 
                "RULE_LABEL_NAMES": ["RPV1_label"]
            }
        ],
    },
}


# The class now inherits from the core scaffolder
class ImageAnalysisScaffolder(ConfigScaffolder):
    """
    Extends the core scaffolder to create fully-populated configuration files
    for specific image analysis workflow recipes.
    """
    
    _COMPONENT_SCHEMATICS = {
        "lv_outflow": _LV_OUTFLOW_SCHEMATIC,
        "aortic_wall": _AORTIC_WALL_SCHEMATIC,
        "pulmonary_artery": _PULMONARY_ARTERY_SCHEMATIC,
        "rv_myocardium": _RV_MYOCARDIUM_SCHEMATIC,
        "la_myocardium": _LA_MYOCARDIUM_SCHEMATIC,
        "ra_myocardium": _RA_MYOCARDIUM_SCHEMATIC,
    }

    def scaffold_components(
        self,
        output_dir: Union[str, Path],
        component_names: list[str],
        overwrite: bool = False,
    ) -> None:
        """
        Generates a set of config files from a list of component schematics.
        
        Args:
            output_dir: The directory where configuration files will be saved.
            component_names: A list of component names (e.g., ["aortic_wall"]).
            overwrite: If True, will overwrite existing files.
        """
        output_dir = Path(output_dir)
        merged_labels = {}
        merged_parameters = {}

        # --- 1. Collect and merge data ---
        for name in component_names:
            schematic = self._COMPONENT_SCHEMATICS.get(name)
            if not schematic:
                raise ValueError(f"Unknown component schematic requested: '{name}'")

            merged_labels.update(schematic["labels"])
            merged_parameters.update(schematic["parameters"])

            # --- 2. Write the unique semantic map for this component ---
            maps_dir = output_dir / "semantic_maps"
            json_map = {k.name: v for k, v in schematic["semantic_map"].items()}
            map_content = json.dumps(json_map, indent=2)
            # REUSE the inherited _write_file method for robustness
            self._write_file(maps_dir / f"{name}.json", map_content, overwrite)

        # --- 3. Write the merged configuration files ---
        # Programmatically generate the full content
        labels_content = yaml.dump({"labels": merged_labels}, sort_keys=False)
        params_content = json.dumps(merged_parameters, indent=2)

        # REUSE the inherited _write_file method
        self._write_file(output_dir / "labels.yaml", labels_content, overwrite)
        self._write_file(output_dir / "parameters.json", params_content, overwrite)
            
        print(f"Scaffolding complete in '{output_dir.resolve()}' for: {', '.join(component_names)}")

    def create_labels_manifest(self, *args, **kwargs):
        """Override the base method to prevent confusion."""
        raise NotImplementedError(
            "Use scaffold_components() to generate a populated labels.yaml. "
            "Generating a template-based manifest is not supported by this scaffolder."
        )