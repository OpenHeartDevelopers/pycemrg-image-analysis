# src/pycemrg_image_analysis/schematics/rings.py

from pycemrg_image_analysis.logic.constants import RingSemanticRole

RING_SCHEMATICS = {
    "lpv1_ring": {
        "labels": {
            "LPV1_label": 8,
            "LPV1_ring_label": 208,
            "LA_myo_label": 104,
        },
        "parameters": {"ring_thickness": 2.0},
        "semantic_map": {
            RingSemanticRole.SOURCE_VEIN_LABEL_NAME: "LPV1_label",
            RingSemanticRole.TARGET_RING_LABEL_NAME: "LPV1_ring_label",
            RingSemanticRole.RING_THICKNESS_PARAMETER_NAME: "ring_thickness",
            RingSemanticRole.ATRIUM_MYOCARDIUM_NAME: "LA_myo_label",
            RingSemanticRole.APPLICATION_STEPS: [],
        },
    },
    
    "lpv2_ring": {
        "labels": {
            "LPV2_label": 9,
            "LPV2_ring_label": 209,
            "LA_myo_label": 104,
        },
        "parameters": {"ring_thickness": 2.0},
        "semantic_map": {
            RingSemanticRole.SOURCE_VEIN_LABEL_NAME: "LPV2_label",
            RingSemanticRole.TARGET_RING_LABEL_NAME: "LPV2_ring_label",
            RingSemanticRole.RING_THICKNESS_PARAMETER_NAME: "ring_thickness",
            RingSemanticRole.ATRIUM_MYOCARDIUM_NAME: "LA_myo_label",
            RingSemanticRole.APPLICATION_STEPS: [],
        },
    },
    
    "rpv1_ring": {
        "labels": {
            "RPV1_label": 10,
            "RPV1_ring_label": 210,
            "LA_myo_label": 104,
            "SVC_label": 13,
        },
        "parameters": {"ring_thickness": 2.0},
        "semantic_map": {
            RingSemanticRole.SOURCE_VEIN_LABEL_NAME: "RPV1_label",
            RingSemanticRole.TARGET_RING_LABEL_NAME: "RPV1_ring_label",
            RingSemanticRole.RING_THICKNESS_PARAMETER_NAME: "ring_thickness",
            RingSemanticRole.ATRIUM_MYOCARDIUM_NAME: "LA_myo_label",
            RingSemanticRole.APPLICATION_STEPS: [
                {"MODE": "REPLACE_ONLY", "RULE_LABEL_NAMES": ["SVC_label"]}
            ],
        },
    },
    
    "rpv2_ring": {
        "labels": {
            "RPV2_label": 11,
            "RPV2_ring_label": 211,
            "LA_myo_label": 104,
        },
        "parameters": {"ring_thickness": 2.0},
        "semantic_map": {
            RingSemanticRole.SOURCE_VEIN_LABEL_NAME: "RPV2_label",
            RingSemanticRole.TARGET_RING_LABEL_NAME: "RPV2_ring_label",
            RingSemanticRole.RING_THICKNESS_PARAMETER_NAME: "ring_thickness",
            RingSemanticRole.ATRIUM_MYOCARDIUM_NAME: "LA_myo_label",
            RingSemanticRole.APPLICATION_STEPS: [],
        },
    },
    
    "laa_ring": {
        "labels": {
            "LAA_label": 12,
            "LAA_ring_label": 212,
            "LA_myo_label": 104,
        },
        "parameters": {"ring_thickness": 2.0},
        "semantic_map": {
            RingSemanticRole.SOURCE_VEIN_LABEL_NAME: "LAA_label",
            RingSemanticRole.TARGET_RING_LABEL_NAME: "LAA_ring_label",
            RingSemanticRole.RING_THICKNESS_PARAMETER_NAME: "ring_thickness",
            RingSemanticRole.ATRIUM_MYOCARDIUM_NAME: "LA_myo_label",
            RingSemanticRole.APPLICATION_STEPS: [],
        },
    },
    
    "svc_ring": {
        "labels": {
            "SVC_label": 13,
            "SVC_ring_label": 213,
            "RA_myo_label": 105,
            "Ao_wall_label": 106,
            "LA_myo_label": 104,
            "RPV1_ring_label": 210,
            "RPV1_label": 10,
            "RPV2_ring_label": 211,
            "RPV2_label": 11,
        },
        "parameters": {"ring_thickness": 2.0},
        "semantic_map": {
            RingSemanticRole.SOURCE_VEIN_LABEL_NAME: "SVC_label",
            RingSemanticRole.TARGET_RING_LABEL_NAME: "SVC_ring_label",
            RingSemanticRole.RING_THICKNESS_PARAMETER_NAME: "ring_thickness",
            RingSemanticRole.ATRIUM_MYOCARDIUM_NAME: "RA_myo_label",
            RingSemanticRole.APPLICATION_STEPS: [
                {"MODE": "REPLACE_ONLY", "RULE_LABEL_NAMES": ["Ao_wall_label"]},
                {"MODE": "REPLACE_ONLY", "RULE_LABEL_NAMES": ["LA_myo_label"]},
                {"MODE": "REPLACE_ONLY", "RULE_LABEL_NAMES": ["RPV1_ring_label"]},
                {"MODE": "REPLACE_ONLY", "RULE_LABEL_NAMES": ["RPV1_label"]},
                {"MODE": "REPLACE_ONLY", "RULE_LABEL_NAMES": ["RPV2_ring_label"]},
                {"MODE": "REPLACE_ONLY", "RULE_LABEL_NAMES": ["RPV2_label"]},
            ],
        },
    },
    
    "ivc_ring": {
        "labels": {
            "IVC_label": 14,
            "IVC_ring_label": 214,
            "RA_myo_label": 105,
        },
        "parameters": {"ring_thickness": 2.0},
        "semantic_map": {
            RingSemanticRole.SOURCE_VEIN_LABEL_NAME: "IVC_label",
            RingSemanticRole.TARGET_RING_LABEL_NAME: "IVC_ring_label",
            RingSemanticRole.RING_THICKNESS_PARAMETER_NAME: "ring_thickness",
            RingSemanticRole.ATRIUM_MYOCARDIUM_NAME: "RA_myo_label",
            RingSemanticRole.APPLICATION_STEPS: [],
        },
    },
}