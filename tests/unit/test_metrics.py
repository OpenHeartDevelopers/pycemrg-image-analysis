# tests/unit/test_metrics.py

"""
Unit tests for pycemrg_image_analysis.utilities.metrics

Tests cover:
- Shape validation
- Known-value correctness
- Edge cases (identical volumes, zero MSE, etc.)
- Error handling (invalid axes, unknown metrics, etc.)
"""

import pytest
import numpy as np

from pycemrg_image_analysis.utilities.metrics import (
    compute_mse,
    compute_psnr,
    compute_ssim,
    compute_gradient_error,
    compare_volumes,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def simple_volumes():
    """Small volumes for fast testing."""
    np.random.seed(42)
    predicted = np.random.rand(4, 8, 8).astype(np.float32)
    ground_truth = np.random.rand(4, 8, 8).astype(np.float32)
    return predicted, ground_truth


@pytest.fixture
def identical_volumes():
    """Identical volumes for testing PSNR=inf and SSIM=1.0."""
    volume = np.random.rand(4, 8, 8).astype(np.float32)
    return volume, volume.copy()


@pytest.fixture
def constant_volume():
    """Constant volume for testing zero gradient."""
    volume = np.ones((4, 8, 8), dtype=np.float32) * 0.5
    return volume


# ============================================================================
# SHAPE VALIDATION TESTS
# ============================================================================


def test_mse_shape_mismatch():
    """MSE should raise ValueError for mismatched shapes."""
    pred = np.random.rand(4, 8, 8)
    gt = np.random.rand(5, 8, 8)  # Different Z dimension

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_mse(pred, gt)


def test_psnr_shape_mismatch():
    """PSNR should raise ValueError for mismatched shapes."""
    pred = np.random.rand(4, 8, 8)
    gt = np.random.rand(4, 10, 8)  # Different Y dimension

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_psnr(pred, gt)


def test_ssim_shape_mismatch():
    """SSIM should raise ValueError for mismatched shapes."""
    pred = np.random.rand(4, 8, 8)
    gt = np.random.rand(4, 8, 10)  # Different X dimension

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_ssim(pred, gt)


def test_gradient_shape_mismatch():
    """Gradient error should raise ValueError for mismatched shapes."""
    pred = np.random.rand(4, 8, 8)
    gt = np.random.rand(6, 8, 8)

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_gradient_error(pred, gt)


def test_compare_volumes_shape_mismatch():
    """compare_volumes should raise ValueError for mismatched shapes."""
    pred = np.random.rand(4, 8, 8)
    gt = np.random.rand(4, 8, 12)

    with pytest.raises(ValueError, match="Shape mismatch"):
        compare_volumes(pred, gt)


# ============================================================================
# MSE TESTS
# ============================================================================


def test_mse_zeros(identical_volumes):
    """MSE of identical volumes should be exactly 0."""
    pred, gt = identical_volumes
    mse = compute_mse(pred, gt)
    assert mse == 0.0


def test_mse_known_value():
    """MSE should match manual calculation."""
    pred = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    gt = np.array([[[1.5, 2.5], [3.5, 4.5]]])

    # Manual: ((0.5)^2 + (0.5)^2 + (0.5)^2 + (0.5)^2) / 4 = 0.25
    expected_mse = 0.25
    mse = compute_mse(pred, gt)

    assert np.isclose(mse, expected_mse)


def test_mse_positive(simple_volumes):
    """MSE should always be non-negative."""
    pred, gt = simple_volumes
    mse = compute_mse(pred, gt)
    assert mse >= 0.0


def test_mse_float_output(simple_volumes):
    """MSE should return Python float, not numpy scalar."""
    pred, gt = simple_volumes
    mse = compute_mse(pred, gt)
    assert isinstance(mse, float)


# ============================================================================
# PSNR TESTS
# ============================================================================


def test_psnr_identical(identical_volumes):
    """PSNR of identical volumes should be infinity."""
    pred, gt = identical_volumes
    psnr = compute_psnr(pred, gt)
    assert psnr == float("inf")


def test_psnr_known_value():
    """PSNR should match manual calculation."""
    # Create volumes with known MSE
    pred = np.array([[[0.0, 0.0], [0.0, 0.0]]])
    gt = np.array([[[0.1, 0.1], [0.1, 0.1]]])

    # MSE = (0.1^2 * 4) / 4 = 0.01
    # PSNR = 10 * log10(1.0^2 / 0.01) = 10 * log10(100) = 20 dB
    expected_psnr = 20.0
    psnr = compute_psnr(pred, gt, data_range=1.0)

    assert np.isclose(psnr, expected_psnr)


def test_psnr_data_range_scaling():
    """PSNR should scale correctly with data_range."""
    pred = np.array([[[0.0]]])
    gt = np.array([[[1.0]]])

    # With data_range=1.0
    psnr_1 = compute_psnr(pred, gt, data_range=1.0)

    # With data_range=2.0 (should increase by 10*log10(4) = 6.02 dB)
    psnr_2 = compute_psnr(pred, gt, data_range=2.0)

    assert np.isclose(psnr_2 - psnr_1, 6.0206, atol=0.01)


def test_psnr_positive(simple_volumes):
    """PSNR should be positive (or inf) for normalized data."""
    pred, gt = simple_volumes
    psnr = compute_psnr(pred, gt)
    assert psnr > 0.0 or psnr == float("inf")


def test_psnr_float_output(simple_volumes):
    """PSNR should return Python float."""
    pred, gt = simple_volumes
    psnr = compute_psnr(pred, gt)
    assert isinstance(psnr, float)


# ============================================================================
# SSIM TESTS
# ============================================================================


def test_ssim_identical(identical_volumes):
    """SSIM of identical volumes should be 1.0."""
    pred, gt = identical_volumes
    ssim = compute_ssim(pred, gt)
    assert np.isclose(ssim, 1.0)


def test_ssim_range(simple_volumes):
    """SSIM should be in [0, 1] range."""
    pred, gt = simple_volumes
    ssim = compute_ssim(pred, gt)
    assert 0.0 <= ssim <= 1.0


def test_ssim_winsize_validation():
    """SSIM should raise ValueError if win_size > min dimension."""
    pred = np.random.rand(4, 8, 8)
    gt = np.random.rand(4, 8, 8)

    # Minimum dimension is 4, win_size=7 should fail
    with pytest.raises(ValueError, match="win_size.*cannot be larger"):
        compute_ssim(pred, gt, win_size=7)


def test_ssim_custom_winsize():
    """SSIM should accept custom win_size."""
    pred = np.random.rand(10, 20, 20)
    gt = np.random.rand(10, 20, 20)

    ssim = compute_ssim(pred, gt, win_size=3)
    assert 0.0 <= ssim <= 1.0


def test_ssim_kwargs_forwarding():
    """SSIM should forward additional kwargs to skimage."""
    pred = np.random.rand(10, 20, 20)
    gt = np.random.rand(10, 20, 20)

    # Test that gaussian_weights parameter is accepted
    ssim = compute_ssim(pred, gt, gaussian_weights=True)
    assert isinstance(ssim, float)


def test_ssim_float_output(simple_volumes):
    """SSIM should return Python float."""
    pred, gt = simple_volumes
    ssim = compute_ssim(pred, gt)
    assert isinstance(ssim, float)


# ============================================================================
# GRADIENT ERROR TESTS
# ============================================================================


def test_gradient_constant_volume(constant_volume):
    """Gradient error should be zero for constant volumes."""
    grad_error = compute_gradient_error(constant_volume, constant_volume, axis=0)
    assert grad_error == 0.0


def test_gradient_different_axes():
    """Gradient error should differ for different axes on non-uniform data."""
    # Create volume with Z-gradient only
    z, y, x = np.meshgrid(
        np.linspace(0, 1, 8),  # Z varies
        np.linspace(0, 0, 16),  # Y constant
        np.linspace(0, 0, 16),  # X constant
        indexing="ij",
    )
    volume_z = z.astype(np.float32)

    pred = volume_z + np.random.randn(*volume_z.shape) * 0.01
    gt = volume_z

    # Z-axis should have higher gradient error than X or Y
    error_z = compute_gradient_error(pred, gt, axis=0)
    error_y = compute_gradient_error(pred, gt, axis=1)
    error_x = compute_gradient_error(pred, gt, axis=2)

    # Z-gradient exists, so error should be non-zero
    assert error_z > 0.0

    # Y and X gradients are near zero, so errors should be smaller
    # (though noise adds some gradient everywhere)
    assert error_z > error_y or error_z > error_x


def test_gradient_invalid_axis():
    """Gradient error should raise ValueError for invalid axis."""
    pred = np.random.rand(4, 8, 8)
    gt = np.random.rand(4, 8, 8)

    with pytest.raises(ValueError, match="axis must be 0, 1, or 2"):
        compute_gradient_error(pred, gt, axis=3)

    with pytest.raises(ValueError, match="axis must be 0, 1, or 2"):
        compute_gradient_error(pred, gt, axis=-1)


def test_gradient_positive(simple_volumes):
    """Gradient error should be non-negative."""
    pred, gt = simple_volumes
    error = compute_gradient_error(pred, gt, axis=0)
    assert error >= 0.0


def test_gradient_float_output(simple_volumes):
    """Gradient error should return Python float."""
    pred, gt = simple_volumes
    error = compute_gradient_error(pred, gt, axis=0)
    assert isinstance(error, float)


# ============================================================================
# BATCH COMPARISON TESTS
# ============================================================================


def test_compare_default_metrics(simple_volumes):
    """compare_volumes should compute default metrics."""
    pred, gt = simple_volumes
    results = compare_volumes(pred, gt)

    # Should have 4 default metrics
    assert set(results.keys()) == {"mse", "psnr", "ssim", "gradient"}

    # All values should be floats
    assert all(isinstance(v, float) for v in results.values())


def test_compare_subset_metrics(simple_volumes):
    """compare_volumes should compute only requested metrics."""
    pred, gt = simple_volumes
    results = compare_volumes(pred, gt, metrics=["mse", "psnr"])

    assert set(results.keys()) == {"mse", "psnr"}


def test_compare_unknown_metric(simple_volumes):
    """compare_volumes should raise ValueError for unknown metrics."""
    pred, gt = simple_volumes

    with pytest.raises(ValueError, match="Unknown metrics"):
        compare_volumes(pred, gt, metrics=["mse", "invalid_metric"])


def test_compare_gradient_aliases(simple_volumes):
    """compare_volumes should support gradient_x/y/z aliases."""
    pred, gt = simple_volumes
    results = compare_volumes(
        pred, gt, metrics=["gradient_x", "gradient_y", "gradient_z"]
    )

    assert set(results.keys()) == {"gradient_x", "gradient_y", "gradient_z"}

    # gradient_z should equal gradient (both are axis=0)
    results_default = compare_volumes(pred, gt, metrics=["gradient"])
    assert np.isclose(results["gradient_z"], results_default["gradient"])


def test_compare_all_metrics(simple_volumes):
    """compare_volumes should handle all available metrics."""
    pred, gt = simple_volumes
    all_metrics = [
        "mse",
        "psnr",
        "ssim",
        "gradient",
        "gradient_x",
        "gradient_y",
        "gradient_z",
    ]

    results = compare_volumes(pred, gt, metrics=all_metrics)

    assert set(results.keys()) == set(all_metrics)
    assert all(isinstance(v, float) for v in results.values())


def test_compare_data_range_forwarding(simple_volumes):
    """compare_volumes should forward data_range to PSNR/SSIM."""
    pred, gt = simple_volumes

    # Compute with different data ranges
    results_1 = compare_volumes(pred, gt, data_range=1.0, metrics=["psnr"])
    results_2 = compare_volumes(pred, gt, data_range=2.0, metrics=["psnr"])

    # PSNR should differ based on data_range
    assert results_1["psnr"] != results_2["psnr"]


def test_compare_empty_metrics(simple_volumes):
    """compare_volumes should handle empty metric list."""
    pred, gt = simple_volumes
    results = compare_volumes(pred, gt, metrics=[])

    assert results == {}


def test_compare_dict_output_type(simple_volumes):
    """compare_volumes should return dictionary."""
    pred, gt = simple_volumes
    results = compare_volumes(pred, gt)

    assert isinstance(results, dict)


# ============================================================================
# INTEGRATION TEST: Full Workflow
# ============================================================================


def test_metrics_workflow_integration():
    """Test complete metrics workflow as interpolation validation would use it."""
    # Simulate interpolation scenario
    np.random.seed(123)
    original = np.random.rand(32, 64, 64)  # Original volume
    interpolated = original + np.random.randn(*original.shape) * 0.05  # With noise

    # Compute all metrics
    results = compare_volumes(
        interpolated,
        original,
        data_range=1.0,
        metrics=["mse", "psnr", "ssim", "gradient"],
    )

    # Sanity checks
    assert 0.0 < results["mse"] < 1.0  # Some error present
    assert 10.0 < results["psnr"] < 50.0  # Reasonable PSNR range
    assert 0.7 < results["ssim"] < 1.0  # High structural similarity
    assert results["gradient"] > 0.0  # Some gradient error

    # All metrics should be valid floats
    assert all(not np.isnan(v) for v in results.values())
    assert all(not np.isinf(v) for v in results.values())
