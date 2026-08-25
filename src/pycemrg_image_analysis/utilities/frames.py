# src/pycemrg_image_analysis/utilities/frames.py
"""Recover the world frame that ``segsmooth`` discards.

``segtools`` is third-party and is not modified. It reads NRRD spacing as::

    if 'spacings' in hdr:
        spacings = hdr.get('spacings')
    else:
        spacings = [hdr['space directions'][0][0],
                    hdr['space directions'][1][1],
                    hdr['space directions'][2][2]]

SimpleITK writes ``space directions`` and no ``spacings`` key, so the fallback
always fires. For an oblique acquisition the diagonal of that matrix is
``spacing * cos(theta)``, carrying a sign — not the spacing. The smoothed
volume is therefore mirrored on every axis whose diagonal entry is negative,
anisotropically stretched by ``true_spacing / |diagonal|``, and stripped of its
rotation (segtools is ``vtkImageData`` throughout, so direction cosines never
enter).

The corruption is a linear map, so it inverts exactly. This module holds that
inverse and nothing else: no file I/O, no filesystem assumptions. Callers read
the two NRRD headers, build a :class:`SmoothFrame`, and get a 4x4 back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

Triple = Tuple[float, float, float]

# meshtools3d multiplies every node coordinate by [meshing] rescaleFactor, whose
# .par default is 1000 -- so mesh points come out in micrometres. Read the real
# value from the parfile rather than trusting this.
DEFAULT_RESCALE_FACTOR = 1000.0


@dataclass(frozen=True)
class SmoothFrame:
    """Geometry captured either side of ``segsmooth``.

    Plain tuples, not arrays: this contract is written to and read from JSON, so
    it has to survive a round trip unchanged.

    The ``resampled_*`` fields describe segsmooth's *input* and carry the only
    surviving record of the rotation. The ``smooth_*`` fields describe its
    *output*, whose header holds ``spacings`` and ``axis mins`` and nothing else.

    Attributes:
        resampled_direction_rows: NRRD ``space directions``, row per axis, LPS.
        resampled_origin: NRRD ``space origin``, LPS mm.
        resampled_spacing: True voxel size of the input, mm.
        resampled_size: Input voxel counts.
        smooth_origin: Output NRRD ``axis mins``, in segsmooth's own frame.
        smooth_spacing: Output NRRD ``spacings``.
        smooth_size: Output voxel counts.
    """

    resampled_direction_rows: Tuple[Triple, Triple, Triple]
    resampled_origin: Triple
    resampled_spacing: Triple
    resampled_size: Tuple[int, int, int]
    smooth_origin: Triple
    smooth_spacing: Triple
    smooth_size: Tuple[int, int, int]

    def direction_matrix(self) -> np.ndarray:
        """``space directions`` as a voxel->world matrix (axis vectors in columns).

        NRRD stores one row per axis; a voxel->world matrix wants those as
        columns, hence the transpose.
        """
        return np.asarray(self.resampled_direction_rows, dtype=float).T


def believed_spacing(frame: SmoothFrame) -> np.ndarray:
    """The signed spacing ``segsmooth`` actually used.

    This is the defect, named in one place: the diagonal of ``space
    directions``. Negative entries mirror their axis; magnitudes below the true
    spacing stretch it.

    Args:
        frame: Geometry captured either side of segsmooth.

    Returns:
        Signed spacing, shape (3,).
    """
    rows = np.asarray(frame.resampled_direction_rows, dtype=float)
    return np.array([rows[0][0], rows[1][1], rows[2][2]], dtype=float)


def distortion_matrix(frame: SmoothFrame) -> np.ndarray:
    """Linear map from segsmooth's frame back to true world (LPS mm).

    Composes the un-mirror and un-stretch (``true / believed``, signed) with the
    rotation that segsmooth dropped.

    Args:
        frame: Geometry captured either side of segsmooth.

    Returns:
        3x3 matrix.

    Raises:
        ValueError: If any believed-spacing entry is zero, which would make the
            corruption non-invertible.
    """
    believed = believed_spacing(frame)
    if np.any(believed == 0.0):
        raise ValueError(
            f"segsmooth read a zero spacing on some axis: {believed.tolist()}. "
            "The diagonal of 'space directions' is degenerate, so the frame "
            "cannot be recovered from these headers."
        )
    return frame.direction_matrix() @ np.diag(1.0 / believed)


def mesh_to_world_affine(
    frame: SmoothFrame,
    rescale_factor: float = DEFAULT_RESCALE_FACTOR,
) -> np.ndarray:
    """4x4 mapping meshtools3d node coordinates to world LPS millimetres.

    The mesh leaves meshtools3d at ``(voxel_index * VX) * rescale_factor``, with
    the origin pinned to the ``(0, 0, 0)`` corner of the ``.inr`` array, because
    the INR header carries neither origin nor orientation. Undoing that means
    three composed steps: back out the rescale, restore segsmooth's origin
    (``axis mins``), then undo the distortion.

    Args:
        frame: Geometry captured either side of segsmooth.
        rescale_factor: ``[meshing] rescaleFactor`` from the meshtools3d
            parfile. Read it from the file; do not assume the default.

    Returns:
        4x4 affine. Input points are in the mesh's own units (micrometres at the
        default rescale factor); output is world LPS millimetres.

    Raises:
        ValueError: If ``rescale_factor`` is zero, or the frame is degenerate.
    """
    if rescale_factor == 0.0:
        raise ValueError("rescale_factor must be non-zero.")

    distortion = distortion_matrix(frame)
    origin = np.asarray(frame.resampled_origin, dtype=float)
    smooth_origin = np.asarray(frame.smooth_origin, dtype=float)

    affine = np.eye(4)
    affine[:3, :3] = distortion / rescale_factor
    affine[:3, 3] = origin + distortion @ (smooth_origin - origin)
    return affine


def smooth_voxel_to_world_affine(frame: SmoothFrame) -> np.ndarray:
    """4x4 mapping smooth-NRRD voxel indices to world LPS millimetres.

    The image-space counterpart of :func:`mesh_to_world_affine`, for checking
    the recovered frame against a reference segmentation without a mesh.

    Args:
        frame: Geometry captured either side of segsmooth.

    Returns:
        4x4 affine.
    """
    distortion = distortion_matrix(frame)
    origin = np.asarray(frame.resampled_origin, dtype=float)
    smooth_origin = np.asarray(frame.smooth_origin, dtype=float)

    affine = np.eye(4)
    affine[:3, :3] = distortion @ np.diag(np.asarray(frame.smooth_spacing, float))
    affine[:3, 3] = origin + distortion @ (smooth_origin - origin)
    return affine


def lps_to_ras(affine: np.ndarray) -> np.ndarray:
    """Convert an LPS-output affine to RAS.

    NRRD and SimpleITK speak LPS; NIfTI on disk and nibabel speak RAS. The two
    differ by the sign of the first two axes.

    Args:
        affine: 4x4 whose output is in LPS.

    Returns:
        New 4x4 whose output is in RAS.
    """
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ np.asarray(affine, dtype=float)


def apply_affine(affine: np.ndarray, points: Sequence[Sequence[float]]) -> np.ndarray:
    """Apply a 4x4 to an (N, 3) point array.

    Args:
        affine: 4x4 transform.
        points: (N, 3) coordinates.

    Returns:
        (N, 3) transformed coordinates.

    Raises:
        ValueError: If ``points`` is not (N, 3).
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {pts.shape}.")
    matrix = np.asarray(affine, dtype=float)
    return (matrix[:3, :3] @ pts.T).T + matrix[:3, 3]


def image_mesh_to_world_affine(
    direction_columns: Sequence[Sequence[float]],
    origin: Triple,
    rescale_factor: float = DEFAULT_RESCALE_FACTOR,
) -> np.ndarray:
    """4x4 for a mesh built straight from an image, with no ``segsmooth`` between.

    Skipping ``smooth`` means skipping the corruption: only the origin and the
    orientation are lost, because the INR format carries neither. There is no
    mirroring or stretch to undo, so this is the simple case.

    Spacing is deliberately not a parameter. A mesh point is
    ``index * spacing * rescale_factor`` and world position is
    ``direction @ (spacing * index) + origin``, so the spacing cancels and only
    the direction cosines and the rescale factor survive.

    Note the convention differs from :class:`SmoothFrame`, which stores NRRD
    ``space directions`` row per axis. Here the axis vectors are **columns**,
    matching ``sitk.Image.GetDirection()`` reshaped to 3x3.

    Args:
        direction_columns: 3x3 direction cosines, axis vectors in columns.
        origin: Image origin, LPS mm.
        rescale_factor: ``[meshing] rescaleFactor`` from the meshtools3d parfile.

    Returns:
        4x4 affine from mesh units to world LPS millimetres.

    Raises:
        ValueError: If ``rescale_factor`` is zero.
    """
    if rescale_factor == 0.0:
        raise ValueError("rescale_factor must be non-zero.")

    affine = np.eye(4)
    affine[:3, :3] = np.asarray(direction_columns, dtype=float) / rescale_factor
    affine[:3, 3] = np.asarray(origin, dtype=float)
    return affine
