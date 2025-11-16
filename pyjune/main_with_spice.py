"""
JunoCam image processing with SPICE kernel and metadata support.

Baby steps version - just loads kernels and metadata properly,
basic image processing without geometric correction.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import spiceypy as spice

from spice_correction import SpiceKernelManager, JunoCamImage


def process_junocam_simple(fname):
    """
    Process JunoCam image - basic channel extraction.

    Args:
        fname: Path to raw JunoCam image

    Returns:
        Tuple of (red, green, blue) channel mosaics
    """
    # Load raw image
    raw = cv2.imread(str(fname), cv2.IMREAD_UNCHANGED)
    if raw is None:
        print(f"Could not open raw image: {fname}")
        sys.exit(1)

    height, width = raw.shape[:2]
    rows = height
    print(f"Raw size: {width} x {rows}")

    # JunoCam parameters
    bandHeight = 128  # strips height for JunoCam visible filters
    bands = 3  # R, G, B filters

    # Calculate number of frames
    frames = rows // (bandHeight * bands)
    print(f"Frames count: {frames}")

    # Filter offsets in the raw image
    # Order: Blue, Green, Red
    redOffset = 2 * bandHeight
    greenOffset = bandHeight
    blueOffset = 0

    # Create output mosaics
    redMosaic = np.zeros((frames * bandHeight, width), dtype=raw.dtype)
    greenMosaic = np.zeros((frames * bandHeight, width), dtype=raw.dtype)
    blueMosaic = np.zeros((frames * bandHeight, width), dtype=raw.dtype)

    # Process each frame (skip first and last - incomplete data)
    for f in range(1, frames - 1):
        baseRow = f * bandHeight * bands

        # Extract strips from raw data
        stripR = raw[baseRow + redOffset : baseRow + redOffset + bandHeight, :]
        stripG = raw[baseRow + greenOffset : baseRow + greenOffset + bandHeight, :]
        stripB = raw[baseRow + blueOffset : baseRow + blueOffset + bandHeight, :]

        # Place into mosaics with proper vertical alignment
        # Each channel has different timing, so different vertical positions
        redMosaic[
            f * bandHeight + bandHeight : (f + 1) * bandHeight + bandHeight, :
        ] = stripR

        greenMosaic[f * bandHeight : (f + 1) * bandHeight, :] = stripG

        blueMosaic[
            f * bandHeight - bandHeight : (f + 1) * bandHeight - bandHeight, :
        ] = stripB

    return redMosaic, greenMosaic, blueMosaic


def main():
    print("=" * 70)
    print("JunoCam Processing with SPICE Support (Baby Steps)")
    print("=" * 70)

    # Initialize SPICE kernels
    print("\n1. Loading SPICE kernels...")
    kernel_manager = SpiceKernelManager()
    kernel_manager.load_kernels()

    try:
        # Load and parse image metadata
        print("\n2. Loading image and metadata...")
        fname = Path("images/raw/JNCE_2021159_34C00080_V01-raw.png")

        # Parse filename and load metadata
        junocam_img = JunoCamImage(fname)

        print("\n3. Image metadata:")
        print(f"   Product ID: {junocam_img.product_id}")
        print(f"   Date: {junocam_img.year}-{junocam_img.doy:03d}")
        print(f"   Orbit: {junocam_img.orbit}")
        print(f"   Filter: {junocam_img.filter_combo}")
        print(f"   Image index: {junocam_img.image_index}")

        if junocam_img.metadata:
            print(f"   Image time: {junocam_img.image_time}")
            print(f"   SCLK: {junocam_img.sclk_string}")

            # Get ephemeris time
            et = junocam_img.get_ephemeris_time()
            utc = spice.et2utc(et, "C", 3)
            print(f"   Ephemeris time: {et:.6f}")
            print(f"   UTC: {utc}")

            # Get spacecraft state
            try:
                state, _ = spice.spkezr("JUNO", et, "J2000", "NONE", "JUPITER")
                range_km = spice.vnorm(state[:3])
                print(f"   Range to Jupiter: {range_km:.1f} km")
            except:
                print(f"   (Could not query spacecraft state)")

        # Process image (basic, no correction yet)
        print("\n4. Processing image...")
        redMosaic, greenMosaic, blueMosaic = process_junocam_simple(fname)

        # Save individual channels
        out_dir = Path("images/processed")
        out_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out_dir / "red_channel.png"), redMosaic)
        cv2.imwrite(str(out_dir / "green_channel.png"), greenMosaic)
        cv2.imwrite(str(out_dir / "blue_channel.png"), blueMosaic)
        print("Single-channel mosaics written.")

        # Create RGB composite
        red8 = cv2.normalize(redMosaic, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        green8 = cv2.normalize(greenMosaic, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        blue8 = cv2.normalize(blueMosaic, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Merge into RGB (OpenCV uses BGR)
        rgbMosaic = cv2.merge([blue8, green8, red8])

        cv2.imwrite(str(out_dir / "combined_rgb.png"), rgbMosaic)
        print("Combined RGB image written.")

        print("\n" + "=" * 70)
        print("✓ Processing complete!")
        print("=" * 70)
        print("\nNext steps:")
        print("  - Check images/processed/ for output")
        print("  - Ready to implement SPICE-based geometric correction")

    finally:
        # Clean up SPICE kernels
        kernel_manager.unload_kernels()


if __name__ == "__main__":
    main()
