# src/pycemrg_image_analysis/utilities/filters.py
import logging 
import SimpleITK as sitk

def distance_map(image: sitk.Image, label: int) -> sitk.Image:
    """
    Generates a Danielsson distance map from a specific label in an image.

    Args:
        image: The input SimpleITK image.
        label: The integer label to calculate the distance from.

    Returns:
        A SimpleITK image representing the distance map.
    """
    # Create a binary mask of the specified label
    binary_mask = sitk.BinaryThreshold(
        image, lowerThreshold=label, upperThreshold=label, insideValue=1, outsideValue=0
    )
    
    distance_map_filter = sitk.DanielssonDistanceMapImageFilter()
    distance_map_filter.InputIsBinaryOn()
    distance_map_filter.UseImageSpacingOff() # We want voxel units for wall thickness
    
    return distance_map_filter.Execute(binary_mask)

def threshold_filter(
    image: sitk.Image, lower: float, upper: float, binarise: bool = False
) -> sitk.Image:
    """
    Applies a threshold filter to an image.

    Args:
        image: The input SimpleITK image.
        lower: The lower threshold value.
        upper: The upper threshold value.
        binarise: If True, creates a binary mask (0s and 1s). If False,
                  sets values outside the range to 0.

    Returns:
        The thresholded SimpleITK image.
    """
    if binarise:
        threshold = sitk.BinaryThresholdImageFilter()
        threshold.SetLowerThreshold(lower)
        threshold.SetUpperThreshold(upper)
        threshold.SetInsideValue(1)
        threshold.SetOutsideValue(0)
    else:
        threshold = sitk.ThresholdImageFilter()
        threshold.SetLower(lower)
        threshold.SetUpper(upper)
        threshold.SetOutsideValue(0)
        
    return threshold.Execute(image)