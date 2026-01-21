# src/pycemrg_image_analysis/logic/builders.py

import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple

from pycemrg.data.labels import LabelManager
from pycemrg_image_analysis.logic.contracts import CylinderCreationContract
from .contracts import MyocardiumCreationContract


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

class MyocardiumPathBuilder:
    """
    A convenience builder for myocardium creation workflows.

    This builder is initialized with the common components needed for a series
    of myocardium creation steps (label manager, parameters, etc.). It then
    simplifies the creation of the specific contract for each step.
    """

    def __init__(
        self,
        output_dir: Path,
        label_manager: LabelManager,
        parameters: dict[str, float],
        input_image: sitk.Image,
    ):
        self._output_dir = output_dir
        self._label_manager = label_manager
        self._parameters = parameters
        self._input_image = input_image

    def build_creation_contract(
        self, output_name: str
    ) -> MyocardiumCreationContract:
        """
        Builds the contract for a single myocardium creation task.

        The `rule` attribute is intentionally left as None. It is the
        responsibility of the specific MyocardiumLogic method being called
        (e.g., `create_aortic_wall`) to define and insert the correct rule.

        Args:
            output_name: The base name for the output file (e.g., "seg_s3b_aortic_wall").

        Returns:
            A MyocardiumCreationContract ready to be passed to a logic method.
        """
        output_path = self._output_dir / f"{output_name}.nrrd"

        return MyocardiumCreationContract(
            input_image=self._input_image,
            rule=None,  # This is a placeholder; the Logic method will fill it in.
            label_manager=self._label_manager,
            parameters=self._parameters,
            output_path=output_path,
        )