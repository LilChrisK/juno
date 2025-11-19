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
from dataclasses import dataclass

from spice_correction import SpiceKernelManager, JunoCamImage
from utils import create_debug_visualization


@dataclass
class Framelet:
    """Represents a single framelet (color band strip) from JunoCam."""

    frame_number: int
    color: str  # 'red', 'green', or 'blue'
    color_index: int  # 0=blue, 1=green, 2=red
    data: np.ndarray
    et: float = 0.0  # Ephemeris time for this framelet

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]


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


def process_junocam_2(fname):
    """
    Process JunoCam image using framelet objects for clean, scalable reconstruction.

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
    band_height = 128  # strips height for JunoCam visible filters
    bands = 3  # R, G, B filters
    frame_height = band_height * bands

    # Calculate number of frames
    frames = rows // frame_height
    print(f"Frames count: {frames}")

    # Color mapping
    color_map = {0: "blue", 1: "green", 2: "red"}

    framelets_by_color = {"red": [], "green": [], "blue": []}

    for f in range(frames):
        for color_idx in range(bands):
            base_row = f * frame_height + color_idx * band_height
            framelet_data = raw[base_row : base_row + band_height, :]

            framelet = Framelet(
                frame_number=f,
                color=color_map[color_idx],
                color_index=color_idx,
                data=framelet_data.copy(),  # Copy to avoid reference issues
            )
            framelets_by_color[framelet.color].append(framelet)

    total_framelets = sum(len(v) for v in framelets_by_color.values())
    print(
        f"Extracted {total_framelets} framelets (R:{len(framelets_by_color['red'])}, "
        f"G:{len(framelets_by_color['green'])}, B:{len(framelets_by_color['blue'])})"
    )

    # Reconstruct color channel mosaics from framelets
    return _reconstruct_mosaics(
        framelets_by_color, frames, band_height, width, raw.dtype
    )


def _reconstruct_mosaics(
    framelets_by_color: dict, frames: int, band_height: int, width: int, dtype
) -> tuple:
    """
    Reconstruct RGB mosaics from framelet objects grouped by color.

    Args:
        framelets_by_color: Dictionary mapping color names to lists of Framelet objects
        frames: Total number of frames
        band_height: Height of each band
        width: Image width
        dtype: Data type for output arrays

    Returns:
        Tuple of (red, green, blue) channel mosaics
    """
    # Create output mosaics
    red_mosaic = np.zeros((frames * band_height, width), dtype=dtype)
    green_mosaic = np.zeros((frames * band_height, width), dtype=dtype)
    blue_mosaic = np.zeros((frames * band_height, width), dtype=dtype)

    # Process each color channel with spatial offsets to align them
    # The RGB filters are at different physical positions on the sensor,
    # so they image different parts of the scene at the same time
    # Skip first and last frames - incomplete data
    for framelet in framelets_by_color["red"]:
        f = framelet.frame_number
        if f == 0 or f >= frames - 1:
            continue
        # Red filter is furthest down the sensor, so offset forward by 1 band
        red_mosaic[
            f * band_height + band_height : (f + 1) * band_height + band_height, :
        ] = framelet.data

    for framelet in framelets_by_color["green"]:
        f = framelet.frame_number
        if f == 0 or f >= frames - 1:
            continue
        # Green is the middle filter - use as reference
        green_mosaic[f * band_height : (f + 1) * band_height, :] = framelet.data

    for framelet in framelets_by_color["blue"]:
        f = framelet.frame_number
        if f == 0 or f >= frames - 1:
            continue
        # Blue filter is furthest up the sensor, so offset backward by 1 band
        blue_mosaic[
            f * band_height - band_height : (f + 1) * band_height - band_height, :
        ] = framelet.data

    return red_mosaic, green_mosaic, blue_mosaic


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
        # fname = Path("images/raw/JNCE_2021159_34C00080_V01-raw.png")
        # fname = Path("images/raw/JNCE_2021159_34C00055_V01-raw.png")
        fname = Path("images/raw/JNCE_2021159_34C00048_V01-raw.png")
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

        # Create debug visualization
        print("\n4. Creating debug visualization...")
        out_dir = Path("images/processed")
        out_dir.mkdir(parents=True, exist_ok=True)
        create_debug_visualization(fname, out_dir / "debug_frame_structure.png")

        # Process image (basic, no correction yet)
        print("\n5. Processing image...")
        redMosaic, greenMosaic, blueMosaic = process_junocam_2(fname)

        # Save individual channels
        out_dir = Path("images/processed")
        out_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out_dir / "red_channel.png"), redMosaic)
        cv2.imwrite(str(out_dir / "green_channel.png"), greenMosaic)
        cv2.imwrite(str(out_dir / "blue_channel.png"), blueMosaic)
        print("Single-channel mosaics written.")

        # Create RGB composite
        red8 = cv2.normalize(redMosaic, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        green8 = cv2.normalize(greenMosaic, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        blue8 = cv2.normalize(blueMosaic, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )

        # Merge into RGB (OpenCV uses BGR)
        rgbMosaic = cv2.merge([blue8, green8, red8])

        cv2.imwrite(str(out_dir / "combined_rgb.png"), rgbMosaic)
        print("Combined RGB image written.")

        print("\n" + "=" * 70)
        print("Processing complete!")
        print("=" * 70)
        print("\nGenerated files:")
        print(
            "  - debug_frame_structure.png: Annotated raw image showing frame/band layout"
        )
        print(
            "  - red_channel.png, green_channel.png, blue_channel.png: Individual channels"
        )
        print("  - combined_rgb.png: RGB composite")

    finally:
        # Clean up SPICE kernels
        kernel_manager.unload_kernels()


if __name__ == "__main__":
    main()
