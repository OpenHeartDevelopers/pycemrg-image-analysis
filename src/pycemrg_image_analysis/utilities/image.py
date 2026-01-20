# src/pycemrg_image_analysis/utilities/image.py

import SimpleITK as sitk
import numpy as np
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


def calculate_cylinder_mask(
    image_shape: Tuple[int, int, int],
    origin: np.ndarray,
    spacing: np.ndarray,
    points: np.ndarray,
    slicer_radius: float,
    slicer_height: float,
) -> np.ndarray:
    """
    Generates a cylindrical mask based on geometric and physical parameters.

    This is a pure, stateless function refactored from the legacy
    FourChamberProcess.cylinder_process method. It operates solely on input
    data and returns a numpy array, with no knowledge of file paths.

    Args:
        image_shape: The shape of the target image space (e.g., (nx, ny, nz)).
        origin: The physical origin (x, y, z) of the image.
        spacing: The physical spacing between pixels/voxels.
        points: A NumPy array of points defining the plane for the cylinder's center.
        slicer_radius: The radius of the cylinder in physical units.
        slicer_height: The height of the cylinder in physical units.

    Returns:
        A NumPy array of type uint8 with the same shape as image_shape,
        where the cylindrical region is marked with a value of 1.
    """
    # Create the output array, ensuring the shape is in the correct order.
    # Legacy code often used (nx, ny, nz) while numpy uses (nz, ny, nx).
    # We will assume the input image_shape is (nx, ny, nz) and work with that.
    cylinder_mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert voxel-based points to world coordinates
    points_coords = origin + spacing * points

    # Calculate cylinder geometry
    cog = np.mean(points_coords, axis=0)
    v1 = points_coords[1, :] - points_coords[0, :]
    v2 = points_coords[2, :] - points_coords[0, :]
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)

    p1 = cog - normal * slicer_height / 2.0
    p2 = cog + normal * slicer_height / 2.0
    n = p2 - p1

    # Optimize search by defining a bounding box around the cylinder
    cube_size = max(slicer_height, slicer_radius) + (2 * np.max(spacing))
    min_bounds = cog - cube_size / 2.0
    max_bounds = cog + cube_size / 2.0

    # Convert physical bounds to voxel indices
    min_indices = np.maximum(np.floor((min_bounds - origin) / spacing).astype(int), 0)
    max_indices = np.minimum(np.ceil((max_bounds - origin) / spacing).astype(int), image_shape)

    # Iterate only within the bounding box
    for i in range(min_indices[0], max_indices[0]):
        for j in range(min_indices[1], max_indices[1]):
            for k in range(min_indices[2], max_indices[2]):
                test_point = origin + spacing * np.array([i, j, k])
                v_p1_to_test = test_point - p1
                v_p2_to_test = test_point - p2

                # Check if the point is between the two end planes of the cylinder
                if np.dot(v_p1_to_test, n) >= 0 and np.dot(v_p2_to_test, n) <= 0:
                    # Check if the point is within the radius
                    distance_from_axis = np.linalg.norm(
                        np.cross(v_p1_to_test, n)
                    ) / np.linalg.norm(n)
                    if distance_from_axis <= slicer_radius:
                        cylinder_mask[i, j, k] = 1

    return cylinder_mask