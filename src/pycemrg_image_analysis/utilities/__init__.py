# src/pycemrg_image_analysis/utilities/__init__.py

from pycemrg_image_analysis.utilities.io import load_image, save_image
from pycemrg_image_analysis.utilities.spatial import (
    compute_target_shape,
    compute_actual_spacing,
    resample_to_isotropic,
)
from pycemrg_image_analysis.utilities.geometry import calculate_cylinder_mask
from pycemrg_image_analysis.utilities.masks import (
    MaskOperationMode,
    add_masks,
    add_masks_replace,
    add_masks_replace_except,
    add_masks_replace_only,
)
from pycemrg_image_analysis.utilities.filters import (
    and_filter,
    distance_map,
    threshold_filter,
)
from pycemrg_image_analysis.utilities.dispatchers import get_mask_operation_dispatcher

from pycemrg_image_analysis.utilities.intensity import (
    clip_intensities,
    normalize_min_max,
    normalize_percentile,
)

from pycemrg_image_analysis.utilities.artifact_simulation import downsample_volume

from pycemrg_image_analysis.utilities.sampling import (
    extract_center_patch,
    extract_random_patch,
)

from pycemrg_image_analysis.utilities.metrics import (
    compute_mse,
    compute_psnr,
    compute_ssim,
    compute_gradient_error,
    compare_volumes,
)

from pycemrg_image_analysis.utilities.augmentation import (
    augment_brightness,
    augment_contrast,
    augment_noise,
    create_slice_shifted_volumes,
)


__all__ = [
    # IO Utilities
    "load_image",
    "save_image",
    # Geometry Utilities
    "calculate_cylinder_mask",
    # Mask Utilities
    "MaskOperationMode",
    "add_masks",
    "add_masks_replace",
    "add_masks_replace_except",
    "add_masks_replace_only",
    # Filter Utilities
    "and_filter",
    "distance_map",
    "threshold_filter",
    # Spatial Utilities
    "compute_target_shape",
    "compute_actual_spacing",
    "resample_to_isotropic",
    # Dispatchers
    "get_mask_operation_dispatcher",
    # Intensity Utilities
    "clip_intensities",
    "normalize_min_max",
    "normalize_percentile",
    # Artifact Simulation
    "downsample_volume",
    # Sampling Utilities
    "extract_center_patch",
    "extract_random_patch",
    # Metrics
    "compute_mse",
    "compute_psnr",
    "compute_ssim",
    "compute_gradient_error",
    "compare_volumes",
    # Augmentation Utilities
    "augment_brightness",
    "augment_contrast",
    "augment_noise",
    "create_slice_shifted_volumes",
]
