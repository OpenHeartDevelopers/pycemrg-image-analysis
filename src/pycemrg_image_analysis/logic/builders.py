# src/pycemrg_image_analysis/logic/builders.py

from pathlib import Path
from typing import Tuple
import numpy as np

from pycemrg_image_analysis.logic.contracts import CylinderCreationContract


class SegmentationPathBuilder:
    """
    A convenience builder to construct path and data contracts.

    This class is designed to be used by an orchestrator. It is initialized
    with high-level, reusable information (like a base output directory and
    image geometry). It then provides methods to build specific, detailed
    contracts for the logic layer, abstracting away path manipulation and
    repetitive data entry.
    """

    def __init__(
        self,
        output_dir: Path,
        origin: np.ndarray,
        spacing: np.ndarray,
        image_shape: Tuple[int, int, int],
    ):
        self._output_dir = output_dir
        self._origin = origin
        self._spacing = spacing
        self._image_shape = image_shape

    def build_cylinder_contract(
        self,
        cylinder_name: str,
        points: np.ndarray,
        slicer_radius: float,
        slicer_height: float,
    ) -> CylinderCreationContract:
        """
        Builds the contract for a single cylinder creation task.

        Args:
            cylinder_name: The base name for the output cylinder file (e.g., "SVC").
            points: NumPy array of points defining the cylinder's plane.
            slicer_radius: The radius of the cylinder in physical units.
            slicer_height: The height of the cylinder in physical units.

        Returns:
            A fully populated CylinderCreationContract.
        """
        output_path = self._output_dir / f"{cylinder_name}.nrrd"

        return CylinderCreationContract(
            image_shape=self._image_shape,
            origin=self._origin,
            spacing=self._spacing,
            points=points,
            slicer_radius=slicer_radius,
            slicer_height=slicer_height,
            output_path=output_path,
        )