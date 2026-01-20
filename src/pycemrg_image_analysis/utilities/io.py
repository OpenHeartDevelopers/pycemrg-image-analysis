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