"""
JunoCam image processing with SPICE-based geometric correction.

This version integrates spacecraft motion correction using SPICE kernels.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
from scipy import ndimage

from spice_correction import SpiceKernelManager, JunoCamImage


def apply_geometric_correction(strip, dx, dy):
    """
    Apply geometric correction to a single image strip.

    Args:
        strip: Image strip as numpy array
        dx: Horizontal pixel offset
        dy: Vertical pixel offset

    Returns:
        Corrected strip
    """
    # Use affine transformation to shift the image
    # Create translation matrix
    rows, cols = strip.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])

    # Apply transformation
    corrected = cv2.warpAffine(strip, M, (cols, rows), flags=cv2.INTER_LINEAR)

    return corrected


def process_junocam_with_spice(fname, kernel_manager):
    """
    Process JunoCam image with SPICE-based geometric correction.

    Args:
        fname: Path to raw JunoCam image
        kernel_manager: Initialized SpiceKernelManager

    Returns:
        Tuple of (red, green, blue) corrected channel mosaics
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

    # Initialize SPICE correction
    junocam_img = JunoCamImage(Path(fname).name)

    # Get pixel offsets from SPICE
    print("Calculating SPICE-based pixel offsets...")
    try:
        pixel_offsets = junocam_img.calculate_pixel_offsets(
            band_height=bandHeight,
            num_frames=frames
        )
        use_spice = True
        print("SPICE correction enabled")
    except Exception as e:
        print(f"Warning: Could not calculate SPICE offsets: {e}")
        print("Falling back to no correction")
        use_spice = False
        pixel_offsets = None

    # Filter offsets in the raw image
    # Assuming order: Blue, Green, Red
    redOffset = 2 * bandHeight
    greenOffset = bandHeight
    blueOffset = 0

    # Create output mosaics
    redMosaic = np.zeros((frames * bandHeight, width), dtype=raw.dtype)
    greenMosaic = np.zeros((frames * bandHeight, width), dtype=raw.dtype)
    blueMosaic = np.zeros((frames * bandHeight, width), dtype=raw.dtype)

    # Process each frame
    for f in range(frames):
        baseRow = f * bandHeight * bands
        outRow = f * bandHeight

        # Get SPICE offsets for this frame (if available)
        if use_spice and f in pixel_offsets:
            red_dx, red_dy = pixel_offsets[f]['RED']
            green_dx, green_dy = pixel_offsets[f]['GREEN']
            blue_dx, blue_dy = pixel_offsets[f]['BLUE']
        else:
            # No correction
            red_dx = red_dy = 0
            green_dx = green_dy = 0
            blue_dx = blue_dy = 0

        # Extract strips
        stripR = raw[baseRow + redOffset : baseRow + redOffset + bandHeight, :]
        stripG = raw[baseRow + greenOffset : baseRow + greenOffset + bandHeight, :]
        stripB = raw[baseRow + blueOffset : baseRow + blueOffset + bandHeight, :]

        # Apply geometric correction
        if use_spice:
            stripR = apply_geometric_correction(stripR, red_dx, red_dy)
            stripG = apply_geometric_correction(stripG, green_dx, green_dy)
            stripB = apply_geometric_correction(stripB, blue_dx, blue_dy)

        # Place into mosaics
        redMosaic[outRow : outRow + bandHeight, :] = stripR
        greenMosaic[outRow : outRow + bandHeight, :] = stripG
        blueMosaic[outRow : outRow + bandHeight, :] = stripB

    return redMosaic, greenMosaic, blueMosaic


def main():
    cwd = Path.cwd()
    print(f"Working directory: {cwd}")

    # Initialize SPICE kernels
    kernel_manager = SpiceKernelManager()
    kernel_manager.load_kernels()

    try:
        # Process image
        fname = Path("images/raw/JNCE_2022056_40C00036_V01-raw.png")

        redMosaic, greenMosaic, blueMosaic = process_junocam_with_spice(
            fname, kernel_manager
        )

        # Save individual channels
        out_dir = Path("images/processed")
        out_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out_dir / "red_channel_spice.png"), redMosaic)
        cv2.imwrite(str(out_dir / "green_channel_spice.png"), greenMosaic)
        cv2.imwrite(str(out_dir / "blue_channel_spice.png"), blueMosaic)
        print("SPICE-corrected single-channel mosaics written.")

        # Create RGB composite
        red8 = cv2.normalize(redMosaic, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        green8 = cv2.normalize(greenMosaic, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        blue8 = cv2.normalize(blueMosaic, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Merge into RGB (OpenCV uses BGR)
        rgbMosaic = cv2.merge([blue8, green8, red8])

        cv2.imwrite(str(out_dir / "combined_rgb_spice.png"), rgbMosaic)
        print("SPICE-corrected combined RGB image written.")

    finally:
        # Clean up SPICE kernels
        kernel_manager.unload_kernels()


if __name__ == "__main__":
    main()
