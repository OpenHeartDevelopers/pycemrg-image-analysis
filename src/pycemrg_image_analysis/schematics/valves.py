# src/pycemrg_image_analysis/schematics/valves.py

from pycemrg_image_analysis.logic.constants import ValveSemanticRole

VALVE_SCHEMATICS = {
    "mitral_valve": {
        "labels": {
            "LA_BP_label": 4,
            "LV_BP_label": 1,
            "MV_label": 201,
        },
        "parameters": {"valve_WT": 4.0},
        "semantic_map": {
            ValveSemanticRole.STRUCTURE_A_NAME: "LA_BP_label",
            ValveSemanticRole.STRUCTURE_B_NAME: "LV_BP_label",
            ValveSemanticRole.TARGET_VALVE_NAME: "MV_label",
            ValveSemanticRole.INTERSECTION_THICKNESS_PARAMETER_NAME: "valve_WT",
            ValveSemanticRole.APPLICATION_STEPS: [
                {"MODE": "REPLACE", "RULE_LABEL_NAMES": []}
            ],
        },
    },
    
    "aortic_valve": {
        "labels": {
            "Ao_BP_label": 6,
            "LV_BP_label": 1,
            "AV_label": 203,
        },
        "parameters": {"valve_WT": 4.0},
        "semantic_map": {
            ValveSemanticRole.STRUCTURE_A_NAME: "Ao_BP_label",
            ValveSemanticRole.STRUCTURE_B_NAME: "LV_BP_label",
            ValveSemanticRole.TARGET_VALVE_NAME: "AV_label",
            ValveSemanticRole.INTERSECTION_THICKNESS_PARAMETER_NAME: "valve_WT",
            ValveSemanticRole.APPLICATION_STEPS: [
                {"MODE": "REPLACE", "RULE_LABEL_NAMES": []}
            ],
        },
    },
    
    "pulmonary_valve": {
        "labels": {
            "PArt_BP_label": 7,
            "RV_BP_label": 3,
            "PV_label": 204,
        },
        "parameters": {"valve_WT": 4.0},
        "semantic_map": {
            ValveSemanticRole.STRUCTURE_A_NAME: "PArt_BP_label",
            ValveSemanticRole.STRUCTURE_B_NAME: "RV_BP_label",
            ValveSemanticRole.TARGET_VALVE_NAME: "PV_label",
            ValveSemanticRole.INTERSECTION_THICKNESS_PARAMETER_NAME: "valve_WT",
            ValveSemanticRole.APPLICATION_STEPS: [
                {"MODE": "REPLACE", "RULE_LABEL_NAMES": []}
            ],
        },
    },
    
    "tricuspid_valve": {
        "labels": {
            "RA_BP_label": 5,
            "RV_BP_label": 3,
            "TV_label": 202,
        },
        "parameters": {"valve_WT": 4.0},
        "semantic_map": {
            ValveSemanticRole.STRUCTURE_A_NAME: "RA_BP_label",
            ValveSemanticRole.STRUCTURE_B_NAME: "RV_BP_label",
            ValveSemanticRole.TARGET_VALVE_NAME: "TV_label",
            ValveSemanticRole.INTERSECTION_THICKNESS_PARAMETER_NAME: "valve_WT",
            ValveSemanticRole.APPLICATION_STEPS: [
                {"MODE": "REPLACE", "RULE_LABEL_NAMES": []}
            ],
        },
    },
}