# src/pycemrg_image_analysis/utilities/io.py

import logging 

import numpy as np
import SimpleITK as sitk

from pathlib import Path
from typing import Tuple

def load_image(image_path: Path) -> sitk.Image:
    """
    Loads an image file using SimpleITK.

    Args:
        image_path: The path to the image file.

    Returns:
        A SimpleITK Image object.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found at: {image_path}")
    return sitk.ReadImage(str(image_path))


def save_image(image: sitk.Image, output_path: Path) -> None:
    """
    Saves a SimpleITK Image object to a file.

    Args:
        image: The SimpleITK Image to save.
        output_path: The path where the image will be saved.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path))

def array_to_image(
    array: np.ndarray,
    origin: Tuple[float, float, float],
    spacing: Tuple[float, float, float],
    numpy_is_xyz: bool = False,
) -> sitk.Image:
    """
    Converts a NumPy array into a SimpleITK Image with specified metadata.

    Handles the common NumPy(z,y,x) vs. SITK(x,y,z) axis ordering issue.

    Args:
        array: The NumPy array to convert.
        origin: The origin of the image in physical space.
        spacing: The spacing of the image in physical space.
        numpy_is_xyz: If True, the input NumPy array is assumed to be in
                      (x, y, z) order and will be transposed. If False
                      (default), it is assumed to be in (z, y, x) order.

    Returns:
        A SimpleITK.Image object.
    """
    if numpy_is_xyz:
        # Transpose from (x, y, z) to the (z, y, x) order SITK expects
        array = np.transpose(array, (2, 1, 0))
    
    image = sitk.GetImageFromArray(array)
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    return image


def save_image_from_array(
    array: np.ndarray,
    origin: Tuple[float, float, float],
    spacing: Tuple[float, float, float],
    output_path: Path,
    numpy_is_xyz: bool = False,
) -> None:
    """
    Saves a NumPy array as a SimpleITK Image file.

    This is a convenience wrapper around array_to_image() and save_image().

    Args:
        array: The NumPy array to save.
        origin: The origin of the image.
        spacing: The spacing of the image.
        output_path: The path where the image will be saved.
        numpy_is_xyz: If True, the input NumPy array is assumed to be in
                      (x, y, z) order. See array_to_image() for details.
    """
    image = array_to_image(array, origin, spacing, numpy_is_xyz=numpy_is_xyz)
    save_image(image, output_path)