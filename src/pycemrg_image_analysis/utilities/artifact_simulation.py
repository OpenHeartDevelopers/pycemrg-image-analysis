# src/pycemrg_image_analysis/utilities/artifact_simulation.py

from typing import Tuple
import SimpleITK as sitk
import numpy as np

def downsample_volume(
    image: sitk.Image,
    target_spacing: Tuple[float, float, float],
    interpolator: int = sitk.sitkLinear,
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

    Returns:
        A new, lower-resolution SimpleITK image.
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
    new_spacing = np.array(target_spacing)
    
    # The new size is the ratio of spacings scaled by the original size.
    # Must be rounded to integer pixels.
    new_size = original_size * (original_spacing / new_spacing)
    new_size = np.ceil(new_size).astype(int).tolist()
    resampler.SetSize(new_size)

    # --- 3. Execute the resampling ---
    return resampler.Execute(image)