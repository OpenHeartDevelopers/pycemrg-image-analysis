# src/pycemrg_image_analysis/schematics/myocardium.py

from pycemrg_image_analysis.logic.constants import MyocardiumSemanticRole

MYOCARDIUM_SCHEMATICS = {
    "lv_outflow": {
        "labels": {"LV_BP_label": 1, "LV_myo_label": 2},
        "parameters": {"LV_neck_WT": 2.0},
        "semantic_map": {
            MyocardiumSemanticRole.SOURCE_BLOOD_POOL_NAME: "LV_BP_label",
            MyocardiumSemanticRole.TARGET_MYOCARDIUM_NAME: "LV_myo_label",
            MyocardiumSemanticRole.WALL_THICKNESS_PARAMETER_NAME: "LV_neck_WT",
            MyocardiumSemanticRole.APPLICATION_STEPS: [
                {"MODE": "ADD", "RULE_LABEL_NAMES": []}
            ],
        },
    },
    
    "aortic_wall": {
        "labels": {
            "LV_BP_label": 1,
            "LV_myo_label": 2,
            "Ao_BP_label": 6,
            "Ao_wall_label": 106,
        },
        "parameters": {"Ao_WT": 2.0},
        "semantic_map": {
            MyocardiumSemanticRole.SOURCE_BLOOD_POOL_NAME: "Ao_BP_label",
            MyocardiumSemanticRole.TARGET_MYOCARDIUM_NAME: "Ao_wall_label",
            MyocardiumSemanticRole.WALL_THICKNESS_PARAMETER_NAME: "Ao_WT",
            MyocardiumSemanticRole.APPLICATION_STEPS: [
                {
                    "MODE": "REPLACE_EXCEPT",
                    "RULE_LABEL_NAMES": ["LV_BP_label", "LV_myo_label", "Ao_BP_label"],
                }
            ],
        },
    },
    
    "pulmonary_artery": {
        "labels": {
            "PArt_BP_label": 7,
            "PArt_wall_label": 107,
            "RV_BP_label": 3,
            "Ao_wall_label": 106,
            "Ao_BP_label": 6,
        },
        "parameters": {"PArt_WT": 2.0},
        "semantic_map": {
            MyocardiumSemanticRole.SOURCE_BLOOD_POOL_NAME: "PArt_BP_label",
            MyocardiumSemanticRole.TARGET_MYOCARDIUM_NAME: "PArt_wall_label",
            MyocardiumSemanticRole.WALL_THICKNESS_PARAMETER_NAME: "PArt_WT",
            MyocardiumSemanticRole.APPLICATION_STEPS: [
                {
                    "MODE": "REPLACE_EXCEPT",
                    "RULE_LABEL_NAMES": ["RV_BP_label", "Ao_wall_label", "Ao_BP_label"],
                }
            ],
        },
    },
    
    "rv_myocardium": {
        "labels": {
            "RV_BP_label": 3,
            "RV_myo_label": 103,
            "Ao_wall_label": 106,
        },
        "parameters": {"RV_WT": 3.5},
        "semantic_map": {
            MyocardiumSemanticRole.SOURCE_BLOOD_POOL_NAME: "RV_BP_label",
            MyocardiumSemanticRole.TARGET_MYOCARDIUM_NAME: "RV_myo_label",
            MyocardiumSemanticRole.WALL_THICKNESS_PARAMETER_NAME: "RV_WT",
            MyocardiumSemanticRole.APPLICATION_STEPS: [
                {"MODE": "REPLACE_ONLY", "RULE_LABEL_NAMES": ["Ao_wall_label"]}
            ],
        },
    },
    
    "la_myocardium": {
        "labels": {
            "LA_BP_label": 4,
            "LA_myo_label": 104,
            "RA_BP_label": 5,
            "SVC_label": 13,
        },
        "parameters": {"LA_WT": 2.0},
        "semantic_map": {
            MyocardiumSemanticRole.SOURCE_BLOOD_POOL_NAME: "LA_BP_label",
            MyocardiumSemanticRole.TARGET_MYOCARDIUM_NAME: "LA_myo_label",
            MyocardiumSemanticRole.WALL_THICKNESS_PARAMETER_NAME: "LA_WT",
            MyocardiumSemanticRole.APPLICATION_STEPS: [
                {"MODE": "REPLACE_ONLY", "RULE_LABEL_NAMES": ["RA_BP_label"]},
                {"MODE": "REPLACE_ONLY", "RULE_LABEL_NAMES": ["SVC_label"]},
            ],
        },
    },
    
    "ra_myocardium": {
        "labels": {"RA_BP_label": 5, "RA_myo_label": 105, "RPV1_label": 10},
        "parameters": {"RA_WT": 2.0},
        "semantic_map": {
            MyocardiumSemanticRole.SOURCE_BLOOD_POOL_NAME: "RA_BP_label",
            MyocardiumSemanticRole.TARGET_MYOCARDIUM_NAME: "RA_myo_label",
            MyocardiumSemanticRole.WALL_THICKNESS_PARAMETER_NAME: "RA_WT",
            MyocardiumSemanticRole.APPLICATION_STEPS: [
                {"MODE": "REPLACE_ONLY", "RULE_LABEL_NAMES": ["RPV1_label"]}
            ],
        },
    },
    
    "myo_push_steps": {
        "labels": {
            "LA_myo_label": 104,
            "LA_BP_label": 4,
            "Ao_wall_label": 106,
            "PArt_wall_label": 107,
            "PArt_BP_label": 7,
            "LV_myo_label": 2,
            "RV_myo_label": 103,
            "RV_BP_label": 3,
        },
        "parameters": {"LA_WT": 2.0, "PArt_WT": 2.0, "RV_WT": 3.5},
        "semantic_map": {},  # No semantic map for push steps
    },
}