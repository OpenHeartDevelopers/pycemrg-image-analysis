# src/pycemrg_image_analysis/utilities/metrics.py

"""
Image quality metrics for volume comparison and validation.

All functions expect pre-normalized data in [0, 1] range and arrays in (Z, Y, X) format.
"""

import logging
import numpy as np
from typing import List, Dict
from skimage.metrics import structural_similarity

logger = logging.getLogger(__name__)


def _validate_shapes(predicted: np.ndarray, ground_truth: np.ndarray) -> None:
    """
    Validate that two arrays have matching shapes.

    Args:
        predicted: First array
        ground_truth: Second array

    Raises:
        ValueError: If shapes don't match
    """
    if predicted.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted.shape} != ground_truth {ground_truth.shape}"
        )


def compute_mse(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Compute Mean Squared Error between two volumes.

    Args:
        predicted: Predicted volume (any shape)
        ground_truth: Ground truth volume (same shape as predicted)

    Returns:
        MSE value (scalar). Lower is better.

    Raises:
        ValueError: If shapes don't match

    Example:
        >>> pred = np.random.rand(64, 128, 128)
        >>> gt = np.random.rand(64, 128, 128)
        >>> mse = compute_mse(pred, gt)
    """
    _validate_shapes(predicted, ground_truth)
    return float(np.mean((predicted - ground_truth) ** 2))


def compute_psnr(
    predicted: np.ndarray, ground_truth: np.ndarray, data_range: float = 1.0
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two volumes.

    Args:
        predicted: Predicted volume
        ground_truth: Ground truth volume
        data_range: Maximum possible pixel value (1.0 for normalized [0,1] data)

    Returns:
        PSNR in dB (higher is better). Typical range: 20-50 dB for images.

    Raises:
        ValueError: If shapes don't match or MSE is zero

    Note:
        PSNR = 10 * log10(data_range² / MSE)
        Returns inf if volumes are identical (MSE = 0).

    Example:
        >>> pred = np.random.rand(64, 128, 128)
        >>> gt = np.random.rand(64, 128, 128)
        >>> psnr = compute_psnr(pred, gt, data_range=1.0)
    """
    _validate_shapes(predicted, ground_truth)

    mse = compute_mse(predicted, ground_truth)

    if mse == 0:
        logger.warning("MSE is zero - volumes are identical. Returning inf.")
        return float("inf")

    return float(10 * np.log10((data_range**2) / mse))


def compute_ssim(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    data_range: float = 1.0,
    win_size: int = 7,
    channel_axis: int = None,
    **kwargs,
) -> float:
    """
    Compute Structural Similarity Index between two volumes.

    Uses scikit-image's 3D SSIM implementation by default.

    Args:
        predicted: Predicted volume in (Z, Y, X) format
        ground_truth: Ground truth volume in (Z, Y, X) format
        data_range: Maximum possible pixel value (1.0 for normalized data)
        win_size: Window size for local statistics (must be odd). Default 7.
        channel_axis: Channel axis (None for single-channel 3D volumes)
        **kwargs: Additional arguments passed to skimage.metrics.structural_similarity
                  (e.g., gaussian_weights, sigma, use_sample_covariance)

    Returns:
        SSIM value in [0, 1] where 1 indicates perfect structural similarity.
        Typical values: >0.9 is excellent, 0.7-0.9 is good.

    Raises:
        ValueError: If shapes don't match or win_size is too large for volume

    Note:
        - For anisotropic spacing (e.g., 1×1×8mm), consider using smaller win_size
        - The function computes 3D SSIM which preserves inter-slice context

    Example:
        >>> pred = np.random.rand(64, 128, 128)
        >>> gt = np.random.rand(64, 128, 128)
        >>> ssim = compute_ssim(pred, gt, data_range=1.0)
    """
    _validate_shapes(predicted, ground_truth)

    # Validate win_size against smallest dimension
    min_dim = min(predicted.shape)
    if win_size > min_dim:
        raise ValueError(
            f"win_size ({win_size}) cannot be larger than smallest dimension ({min_dim})"
        )

    return float(
        structural_similarity(
            predicted,
            ground_truth,
            data_range=data_range,
            win_size=win_size,
            channel_axis=channel_axis,
            **kwargs,
        )
    )


def compute_gradient_error(
    predicted: np.ndarray, ground_truth: np.ndarray, axis: int = 0
) -> float:
    """
    Compute mean absolute gradient error along specified axis.

    Measures how well edge sharpness is preserved. Particularly useful
    for evaluating interpolation quality along the interpolation axis.

    Args:
        predicted: Predicted volume in (Z, Y, X) format
        ground_truth: Ground truth volume in (Z, Y, X) format
        axis: Axis along which to compute gradient:
              - 0 = Z-axis (depth/slice direction)
              - 1 = Y-axis (height)
              - 2 = X-axis (width)

    Returns:
        Mean absolute gradient difference (scalar). Lower is better.
        Value range depends on data range and content.

    Raises:
        ValueError: If shapes don't match or axis is invalid

    Note:
        Uses numpy.gradient with default spacing (1.0 between points).
        For physical spacing-aware gradients, preprocess with spacing weights.

    Example:
        >>> pred = np.random.rand(64, 128, 128)
        >>> gt = np.random.rand(64, 128, 128)
        >>> # Test Z-axis gradient preservation (interpolation quality)
        >>> z_error = compute_gradient_error(pred, gt, axis=0)
    """
    _validate_shapes(predicted, ground_truth)

    if axis not in [0, 1, 2]:
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}")

    # Compute gradients along specified axis
    grad_pred = np.gradient(predicted, axis=axis)
    grad_gt = np.gradient(ground_truth, axis=axis)

    # Mean absolute error of gradients
    return float(np.mean(np.abs(grad_pred - grad_gt)))


def compare_volumes(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    data_range: float = 1.0,
    metrics: List[str] = None,
) -> Dict[str, float]:
    """
    Compute multiple quality metrics at once for efficient batch comparison.

    Args:
        predicted: Predicted volume in (Z, Y, X) format
        ground_truth: Ground truth volume in (Z, Y, X) format
        data_range: Data range for PSNR/SSIM (1.0 for normalized [0,1] data)
        metrics: List of metric names to compute. Options:
                 - 'mse': Mean Squared Error
                 - 'psnr': Peak Signal-to-Noise Ratio
                 - 'ssim': Structural Similarity Index
                 - 'gradient': Gradient error (Z-axis by default)
                 - 'gradient_x': Gradient error along X-axis
                 - 'gradient_y': Gradient error along Y-axis
                 - 'gradient_z': Gradient error along Z-axis
                 If None, computes all standard metrics: ['mse', 'psnr', 'ssim', 'gradient']

    Returns:
        Dictionary mapping metric names to float values.

    Raises:
        ValueError: If unknown metric name is provided or shapes don't match

    Example:
        >>> pred = np.random.rand(64, 128, 128)
        >>> gt = np.random.rand(64, 128, 128)
        >>> results = compare_volumes(pred, gt, metrics=['mse', 'psnr', 'ssim'])
        >>> print(f"MSE: {results['mse']:.4f}, PSNR: {results['psnr']:.2f} dB")
    """
    _validate_shapes(predicted, ground_truth)

    # Default metric set
    if metrics is None:
        metrics = ["mse", "psnr", "ssim", "gradient"]

    # Metric dispatcher
    metric_functions = {
        "mse": lambda: compute_mse(predicted, ground_truth),
        "psnr": lambda: compute_psnr(predicted, ground_truth, data_range),
        "ssim": lambda: compute_ssim(predicted, ground_truth, data_range),
        "gradient": lambda: compute_gradient_error(predicted, ground_truth, axis=0),
        "gradient_x": lambda: compute_gradient_error(predicted, ground_truth, axis=2),
        "gradient_y": lambda: compute_gradient_error(predicted, ground_truth, axis=1),
        "gradient_z": lambda: compute_gradient_error(predicted, ground_truth, axis=0),
    }

    # Validate metric names
    unknown_metrics = set(metrics) - set(metric_functions.keys())
    if unknown_metrics:
        valid_metrics = ", ".join(metric_functions.keys())
        raise ValueError(
            f"Unknown metrics: {unknown_metrics}. Valid options: {valid_metrics}"
        )

    # Compute requested metrics
    results = {}
    for metric_name in metrics:
        try:
            results[metric_name] = metric_functions[metric_name]()
        except Exception as e:
            logger.error(f"Failed to compute {metric_name}: {e}")
            results[metric_name] = float("nan")

    return results
