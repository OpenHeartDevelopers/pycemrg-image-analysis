#!/usr/bin/env python
# scripts/utilities/inr_to_nifti.py

"""
Convert an INR image file to NIfTI (.nii.gz).

Usage:
    python scripts/utilities/inr_to_nifti.py --input IMAGE.inr --output IMAGE.nii.gz
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pycemrg_image_analysis.utilities.io import convert_inr_to_image, save_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an INR file to NIfTI.")
    parser.add_argument("--input",  required=True, type=Path, metavar="INR",    help="Path to the input .inr file.")
    parser.add_argument("--output", required=True, type=Path, metavar="NIFTI",  help="Path for the output .nii / .nii.gz file.")
    args = parser.parse_args()

    image = convert_inr_to_image(args.input)
    save_image(image, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
