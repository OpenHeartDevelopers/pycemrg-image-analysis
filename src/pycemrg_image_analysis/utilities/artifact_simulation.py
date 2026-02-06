# src/pycemrg_image_analysis/utilities/artifact_simulation.py

import logging
from typing import Tuple
import SimpleITK as sitk
import numpy as np

logger = logging.getLogger(__name__)


def downsample_volume(
    image: sitk.Image,
    target_spacing: Tuple[float, float, float],
    interpolator: int = sitk.sitkLinear,
    preserve_extent: bool = False,
) -> sitk.Image:
    """
    Downsamples a SimpleITK image to a new target spacing.

    This function simulates a lower-resolution acquisition by resampling the
    image grid while preserving the physical space it occupies.

    Args:
        image: The input high-resolution SimpleITK image.
        target_spacing: The desired new spacing in physical units (e.g., mm).
        interpolator: The SimpleITK interpolator to use. Defaults to linear.
                      For masks, sitk.sitkNearestNeighbor should be used.
        preserve_extent: If True, uses (dim-1)×spacing formula to preserve
                        physical extent. Use this when:
                        - Training ML models that normalize coordinates by extent
                        - Encoder-decoder architectures (e.g., FAE)
                        - Physical dimensions are more critical than exact spacing

                        If False (default), uses conservative ceil() calculation:
                        - Guarantees no loss of spatial information
                        - May slightly oversample
                        - Legacy behavior for backward compatibility

    Returns:
        A new, lower-resolution SimpleITK image.

    Note:
        When preserve_extent=True, the actual output spacing may differ slightly
        from target_spacing due to integer rounding of voxel counts. The physical
        extent will be preserved within one voxel of the target spacing.
    """
    if image.GetDimension() != 3 or len(target_spacing) != 3:
        raise ValueError("This function only supports 3D images and spacings.")

    # --- 1. Set up the Resample filter ---
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)

    # --- 2. Define the output grid ---
    # Use the same origin and direction as the input image to maintain
    # physical alignment.
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())

    # Set the desired output spacing
    resampler.SetOutputSpacing(target_spacing)

    # Calculate the new output size to cover the same physical volume
    original_size = np.array(image.GetSize(), dtype=int)
    original_spacing = np.array(image.GetSpacing())
    target_spacing_arr = np.array(target_spacing)

    if preserve_extent:
        # --- EXTENT-PRESERVING CALCULATION ---
        # Physical extent = (dimension - 1) × spacing
        # This is the formula used by coordinate normalization in FAE and similar models
        extents = (original_size - 1) * original_spacing

        # Calculate new size: extent / new_spacing + 1
        # Round to nearest integer - extent takes priority over exact spacing match
        new_size = np.round(extents / target_spacing_arr).astype(int) + 1
        new_size = new_size.tolist()

        # Log extent preservation for debugging
        new_extents = (np.array(new_size) - 1) * target_spacing_arr
        extent_diffs = new_extents - extents

        logger.debug(f"Extent-preserving downsampling:")
        logger.debug(f"  Original extents: {extents}")
        logger.debug(f"  New extents: {new_extents}")
        logger.debug(f"  Differences: {extent_diffs} (should be < target_spacing)")

    else:
        # --- LEGACY CALCULATION ---
        # The new size is the ratio of spacings scaled by the original size.
        # Ceiling ensures we never undersample (legacy conservative behavior)
        new_size = original_size * (original_spacing / target_spacing_arr)
        new_size = np.ceil(new_size).astype(int).tolist()

    resampler.SetSize(new_size)

    # --- 3. Execute the resampling ---
    return resampler.Execute(image)

