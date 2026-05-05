# src/pycemrg_image_analysis/utilities/io.py

import logging 

import numpy as np
import SimpleITK as sitk

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

def array_to_image(
    array: np.ndarray,
    origin: Tuple[float, float, float],
    spacing: Tuple[float, float, float],
    numpy_is_xyz: bool = False,
) -> sitk.Image:
    """
    Converts a NumPy array into a SimpleITK Image with specified metadata.

    Handles the common NumPy(z,y,x) vs. SITK(x,y,z) axis ordering issue.

    Args:
        array: The NumPy array to convert.
        origin: The origin of the image in physical space.
        spacing: The spacing of the image in physical space.
        numpy_is_xyz: If True, the input NumPy array is assumed to be in
                      (x, y, z) order and will be transposed. If False
                      (default), it is assumed to be in (z, y, x) order.

    Returns:
        A SimpleITK.Image object.
    """
    if numpy_is_xyz:
        # Transpose from (x, y, z) to the (z, y, x) order SITK expects
        array = np.transpose(array, (2, 1, 0))
    
    image = sitk.GetImageFromArray(array)
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    return image


_INR_DTYPE_MAP = {
    ("unsigned fixed", 8):  np.uint8,
    ("unsigned fixed", 16): np.uint16,
    ("unsigned fixed", 32): np.uint32,
    ("signed fixed",   8):  np.int8,
    ("signed fixed",   16): np.int16,
    ("signed fixed",   32): np.int32,
    ("float",          32): np.float32,
    ("float",          64): np.float64,
}

_NP_TO_INR_DTYPE = {v: k for k, v in _INR_DTYPE_MAP.items()}


def convert_inr_to_image(inr_path: Path) -> sitk.Image:
    """
    Reads an INR file and returns a SimpleITK Image.

    INR stores voxels in Fortran (x-fastest) order, so the raw buffer is
    reshaped as (xdim, ydim, zdim) and passed to array_to_image with
    numpy_is_xyz=True, which transposes to the (z, y, x) layout that
    sitk.GetImageFromArray expects.

    INR headers carry no origin; origin is fixed at (0, 0, 0).
    """
    if not inr_path.exists():
        raise FileNotFoundError(f"INR file not found at: {inr_path}")

    with open(inr_path, "rb") as fh:
        raw_header = ""
        while True:
            line = fh.readline().decode("utf-8")
            raw_header += line
            if line.strip() == "##}":
                break

        fields = {}
        for line in raw_header.split("\n"):
            if "=" in line:
                key, _, value = line.partition("=")
                fields[key.strip()] = value.strip()

        xdim = int(fields["XDIM"])
        ydim = int(fields["YDIM"])
        zdim = int(fields["ZDIM"])
        spacing = (float(fields["VX"]), float(fields["VY"]), float(fields["VZ"]))
        pixsize = int(fields["PIXSIZE"].split()[0])
        dtype_key = fields["TYPE"]

        np_dtype = _INR_DTYPE_MAP.get((dtype_key, pixsize))
        if np_dtype is None:
            raise ValueError(
                f"Unsupported INR dtype: TYPE={dtype_key!r}, PIXSIZE={pixsize}"
            )

        data = np.frombuffer(fh.read(), dtype=np_dtype)
        data = data.reshape((xdim, ydim, zdim), order="F")

    return array_to_image(data, origin=(0.0, 0.0, 0.0), spacing=spacing, numpy_is_xyz=True)


def convert_image_to_inr(image: sitk.Image, inr_path: Path) -> None:
    """
    Writes a SimpleITK Image to an INR file.

    INR stores voxels in Fortran (x-fastest) order. The array extracted from
    sitk (z, y, x layout) is transposed to (x, y, z) before flattening.

    INR headers carry no origin; the image origin is not written.
    """
    array = sitk.GetArrayFromImage(image)       # (z, y, x)
    array = np.transpose(array, (2, 1, 0))      # → (x, y, z)

    inr_type, pixsize = _NP_TO_INR_DTYPE.get(array.dtype.type, (None, None))
    if inr_type is None:
        raise ValueError(f"Unsupported numpy dtype for INR conversion: {array.dtype}")

    xdim, ydim, zdim = array.shape
    vx, vy, vz = image.GetSpacing()

    header_core = (
        "#INRIMAGE-4#{\n"
        f"XDIM={xdim}\n"
        f"YDIM={ydim}\n"
        f"ZDIM={zdim}\n"
        "VDIM=1\n"
        f"TYPE={inr_type}\n"
        f"PIXSIZE={pixsize} bits\n"
        "SCALE=2**0\n"
        "CPU=decm\n"
        f"VX={vx}\n"
        f"VY={vy}\n"
        f"VZ={vz}\n"
    )

    terminator = "##}\n"
    total = 256
    while total < len(header_core) + len(terminator):
        total += 256
    padding = "\n" * (total - len(header_core) - len(terminator))
    header = (header_core + padding + terminator).encode("utf-8")

    inr_path.parent.mkdir(parents=True, exist_ok=True)
    with open(inr_path, "wb") as fh:
        fh.write(header)
        fh.write(array.tobytes(order="F"))


def save_image_from_array(
    array: np.ndarray,
    origin: Tuple[float, float, float],
    spacing: Tuple[float, float, float],
    output_path: Path,
    numpy_is_xyz: bool = False,
) -> None:
    """
    Saves a NumPy array as a SimpleITK Image file.

    This is a convenience wrapper around array_to_image() and save_image().

    Args:
        array: The NumPy array to save.
        origin: The origin of the image.
        spacing: The spacing of the image.
        output_path: The path where the image will be saved.
        numpy_is_xyz: If True, the input NumPy array is assumed to be in
                      (x, y, z) order. See array_to_image() for details.
    """
    image = array_to_image(array, origin, spacing, numpy_is_xyz=numpy_is_xyz)
    save_image(image, output_path)