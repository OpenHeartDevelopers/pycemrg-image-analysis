# src/pycemrg_image_analysis/logic/segmentation.py

import SimpleITK as sitk
import numpy as np

from pycemrg_image_analysis.logic.contracts import CylinderCreationContract
from pycemrg_image_analysis.utilities.geometry import calculate_cylinder_mask


class SegmentationLogic:
    """
    Provides stateless logic for common segmentation manipulation tasks.

    This class contains the core scientific workflows. It operates on data
    (passed via contracts) and returns data (typically SimpleITK images or
    NumPy arrays). It has no knowledge of file paths or project structures.
    """

    def create_cylinder(self, contract: CylinderCreationContract) -> sitk.Image:
        """
        Creates a cylinder mask based on the provided contract.

        This method orchestrates the cylinder creation process by:
        1. Calling the low-level utility function to perform the calculation.
        2. Converting the resulting array into a SimpleITK image.
        3. Setting the correct physical metadata (origin, spacing) on the image.

        Args:
            contract: A CylinderCreationContract containing all necessary
                      parameters for the operation.

        Returns:
            A SimpleITK.Image object representing the cylinder mask. The image
            will have the correct origin and spacing set.
        """
        # 1. Call the pure utility function to do the heavy lifting
        cylinder_mask_array = calculate_cylinder_mask(
            image_shape=contract.image_shape,
            origin=contract.origin,
            spacing=contract.spacing,
            points=contract.points,
            slicer_radius=contract.slicer_radius,
            slicer_height=contract.slicer_height,
        )

        # 2. Convert the NumPy array result to a SimpleITK image
        # Note: SimpleITK expects the array shape in (z, y, x) order, but our
        # calculation was done in (x, y, z). We need to transpose the array.
        # This is a common point of error we are handling explicitly.
        cylinder_mask_array_transposed = np.transpose(cylinder_mask_array, (2, 1, 0))
        cylinder_image = sitk.GetImageFromArray(cylinder_mask_array_transposed)

        # 3. Set the physical space metadata on the new image
        # converting contract origin/spacing to lists, which SimpleITK expects
        cylinder_image.SetOrigin(contract.origin.tolist())
        cylinder_image.SetSpacing(contract.spacing.tolist())

        return cylinder_image

