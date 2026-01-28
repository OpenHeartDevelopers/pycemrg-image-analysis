import numpy as np
import pytest

from pycemrg_image_analysis.utilities import sampling

@pytest.fixture
def large_volume() -> np.ndarray:
    """A 10x20x30 volume (Z, Y, X)."""
    return np.arange(10 * 20 * 30).reshape((10, 20, 30))

def test_extract_center_patch(large_volume):
    patch_size = (4, 8, 12) # d, h, w
    patch = sampling.extract_center_patch(large_volume, patch_size)
    
    assert patch.shape == patch_size
    
    # Verify the content is from the center
    # Center of volume: (5, 10, 15)
    # Half patch: (2, 4, 6)
    # Start corner: (3, 6, 9)
    # End corner: (7, 14, 21)
    expected_sub_volume = large_volume[3:7, 6:14, 9:21]
    assert np.array_equal(patch, expected_sub_volume)

def test_extract_center_patch_with_padding(large_volume):
    # Patch is larger than volume in the Z dimension
    patch_size = (20, 10, 10) # d, h, w
    patch = sampling.extract_center_patch(
        large_volume, patch_size, pad_mode='constant', constant_values=-1
    )
    middle_value = large_volume[5, 10, 15]
    
    assert patch.shape == patch_size
    # Check that padding was applied (contains -1)
    assert -1 in patch
    # Check that original data is still there
    assert middle_value in patch

def test_extract_random_patch(large_volume):
    patch_size = (4, 8, 12)
    patch = sampling.extract_random_patch(large_volume, patch_size)
    
    assert patch.shape == patch_size
    # A simple check to ensure it's a subset of the original
    assert patch[0,0,0] >= 0
    assert patch[-1,-1,-1] < large_volume.size

def test_extract_random_patch_reproducibility(large_volume):
    patch_size = (4, 8, 12)
    seed = 42
    
    rng1 = np.random.RandomState(seed)
    patch1 = sampling.extract_random_patch(large_volume, patch_size, rng=rng1)
    
    rng2 = np.random.RandomState(seed)
    patch2 = sampling.extract_random_patch(large_volume, patch_size, rng=rng2)
    
    assert np.array_equal(patch1, patch2)

def test_extract_random_patch_with_padding(large_volume):
    patch_size = (20, 10, 10) # Larger in Z
    patch = sampling.extract_random_patch(
        large_volume, patch_size, pad_mode='symmetric'
    )
    
    assert patch.shape == patch_size
    # Check that it's not just zeros (symmetric padding should reflect values)
    assert not np.all(patch[0,:,:] == 0)