import numpy as np
import SimpleITK as sitk
import pytest

from pycemrg_image_analysis.utilities import artifact_simulation

@pytest.fixture
def sample_image() -> sitk.Image:
    """A sample 10x20x30 image with 1x1x1mm spacing."""
    size = [10, 20, 30]
    image = sitk.Image(size, sitk.sitkFloat32)
    image.SetSpacing([1.0, 1.0, 1.0])
    image.SetOrigin([0.0, 0.0, 0.0])
    # Fill with a gradient to test interpolation
    arr = np.arange(np.prod(size), dtype=np.float32).reshape(size[2], size[1], size[0])
    img_from_arr = sitk.GetImageFromArray(arr)
    img_from_arr.CopyInformation(image)
    return img_from_arr

def test_downsample_volume(sample_image):
    target_spacing = (1.0, 1.0, 5.0) # Downsample in Z
    
    downsampled = artifact_simulation.downsample_volume(sample_image, target_spacing)
    
    # Verify new spacing
    assert np.allclose(downsampled.GetSpacing(), target_spacing)
    
    # Verify new size is calculated correctly:
    # Original Z size = 30, Z spacing = 1.0
    # New Z spacing = 5.0. Ratio = 1.0 / 5.0 = 0.2
    # New Z size = 30 * 0.2 = 6
    expected_size = [10, 20, 6]
    assert downsampled.GetSize() == tuple(expected_size)
    
    # Verify physical properties are preserved
    assert np.allclose(downsampled.GetOrigin(), sample_image.GetOrigin())
    assert np.allclose(downsampled.GetDirection(), sample_image.GetDirection())
    
def test_downsample_volume_anisotropic(sample_image):
    target_spacing = (2.0, 4.0, 5.0) # Downsample in all axes
    
    downsampled = artifact_simulation.downsample_volume(sample_image, target_spacing)
    
    assert np.allclose(downsampled.GetSpacing(), target_spacing)
    
    # New X size = 10 * (1/2) = 5
    # New Y size = 20 * (1/4) = 5
    # New Z size = 30 * (1/5) = 6
    expected_size = [5, 5, 6]
    assert downsampled.GetSize() == tuple(expected_size)