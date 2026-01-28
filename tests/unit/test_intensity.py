import numpy as np
import pytest

from pycemrg_image_analysis.utilities import intensity

@pytest.fixture
def sample_volume() -> np.ndarray:
    """A sample volume with a clear range for testing."""
    return np.array([
        [-10, 0, 10],
        [50, 100, 150],
        [200, 250, 300]
    ]).astype(np.float32)

def test_clip_intensities(sample_volume):
    min_val, max_val = 0, 100
    clipped = intensity.clip_intensities(sample_volume, min_val, max_val)
    
    assert clipped.min() == min_val
    assert clipped.max() == max_val
    assert clipped[0, 0] == 0  # Was -10
    assert clipped[2, 2] == 100 # Was 300
    assert clipped[1, 1] == 100 # Was 100 (in range)

def test_normalize_min_max(sample_volume):
    normalized = intensity.normalize_min_max(sample_volume)
    
    assert np.isclose(normalized.min(), 0.0)
    assert np.isclose(normalized.max(), 1.0)
    # Check a middle value: (100 - (-10)) / (300 - (-10)) = 110 / 310
    assert np.isclose(normalized[1, 1], 110.0 / 310.0)
    assert normalized.dtype == np.float32

def test_normalize_percentile(sample_volume):
    # For this volume, 10th percentile is approx -1.0, 90th is approx 255.0
    p_min, p_max = 10.0, 90.0
    normalized = intensity.normalize_percentile(sample_volume, p_min, p_max)
    
    assert np.isclose(normalized.min(), 0.0)
    assert np.isclose(normalized.max(), 1.0)

    middle_value = normalized[1, 1] # Corresponds to original 100
    assert 0.0 < middle_value < 1.0, f"Middle value {middle_value} should be between 0 and 1"

def test_normalize_percentile_invalid_range():
    with pytest.raises(ValueError):
        intensity.normalize_percentile(np.zeros((2,2)), p_min=50, p_max=40)