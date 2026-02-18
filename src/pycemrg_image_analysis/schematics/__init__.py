# src/pycemrg_image_analysis/schematics/__init__.py

from pycemrg_image_analysis.schematics.myocardium import MYOCARDIUM_SCHEMATICS
from pycemrg_image_analysis.schematics.valves import VALVE_SCHEMATICS
from pycemrg_image_analysis.schematics.rings import RING_SCHEMATICS

ALL_SCHEMATICS = {
    **MYOCARDIUM_SCHEMATICS,
    **VALVE_SCHEMATICS,
    **RING_SCHEMATICS,
}

__all__ = ["ALL_SCHEMATICS"]