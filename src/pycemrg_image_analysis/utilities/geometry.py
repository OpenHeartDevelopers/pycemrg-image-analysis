# src/pycemrg_image_analysis/utilities/geometry.py

import logging

import numpy as np
import SimpleITK as sitk

from pathlib import Path
from typing import Tuple


def calculate_cylinder_mask(
    image_shape: Tuple[int, int, int],  # (depth, height, width)
    origin: np.ndarray,
    spacing: np.ndarray,
    points: np.ndarray,
    slicer_radius: float,
    slicer_height: float,
) -> np.ndarray:
    """
    Generates a cylindrical mask based on geometric and physical parameters.

    Assumes a (Z, Y, X) axis convention for the image_shape and output array.

    Args:
        image_shape: The shape of the target image space in (depth, height, width).
        origin: The physical origin (x, y, z) of the image.
        spacing: The physical spacing between pixels/voxels (x, y, z).
        points: A NumPy array of points (3, 3) defining the plane.
        slicer_radius: The radius of the cylinder in physical units.
        slicer_height: The height of the cylinder in physical units.

    Returns:
        A NumPy array of type uint8 with shape (depth, height, width).
    """
    cylinder_mask = np.zeros(image_shape, dtype=np.uint8)
    depth, height, width = image_shape

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

    # Bounding box calculation must now respect the different array/physical ordering
    # Spacing and origin are (x,y,z), but image_shape is (z,y,x)
    physical_shape = np.array([width, height, depth])
    physical_max_bounds = origin + (physical_shape - 1) * spacing
    
    cube_size = max(slicer_height, slicer_radius) + (2 * np.max(spacing))
    min_bounds_phys = np.maximum(cog - cube_size / 2.0, origin)
    max_bounds_phys = np.minimum(cog + cube_size / 2.0, physical_max_bounds)

    min_indices_phys = np.floor((min_bounds_phys - origin) / spacing).astype(int)
    max_indices_phys = np.ceil((max_bounds_phys - origin) / spacing).astype(int)
    
    # Iterate using (x, y, z) physical indices
    for k in range(min_indices_phys[2], max_indices_phys[2]):
        for j in range(min_indices_phys[1], max_indices_phys[1]):
            for i in range(min_indices_phys[0], max_indices_phys[0]):
                test_point = origin + spacing * np.array([i, j, k])
                v_p1_to_test = test_point - p1

                if np.dot(v_p1_to_test, n) >= 0 and np.dot(test_point - p2, n) <= 0:
                    distance_from_axis = np.linalg.norm(np.cross(v_p1_to_test, n)) / np.linalg.norm(n)
                    if distance_from_axis <= slicer_radius:
                        # Assign to the (z, y, x) array using the correct indices
                        cylinder_mask[k, j, i] = 1

    return cylinder_mask