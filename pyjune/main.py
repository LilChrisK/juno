"""
JunoCam pinhole camera view processing with SPICE kernel support.
Main entry point for processing JunoCam images.
"""

import cv2
from pathlib import Path
import spiceypy as spice

from spice_correction import SpiceKernelManager, JunoCamImage
from map_projection import JupiterEllipsoid
from utils import create_debug_visualization
from pinhole_projection import extract_framelets, project_framelets_to_pinhole_view


def main():
    print("=" * 70)
    print("JUNOCAM PINHOLE VIEW PROCESSING")
    print("=" * 70)

    # Load SPICE kernels
    print("\n1. Loading SPICE kernels...")
    km = SpiceKernelManager()
    km.load_kernels()

    try:
        # Initialize Jupiter ellipsoid
        print("\n2. Initializing Jupiter ellipsoid...")
        ellipsoid = JupiterEllipsoid()

        # Load image
        print("\n3. Loading image metadata...")
        fname = Path("images/raw/JNCE_2021159_34C00080_V01-raw.png")
        # fname = Path("images/raw/JNCE_2021159_34C00055_V01-raw.png")
        # fname = Path("images/raw/JNCE_2021159_34C00048_V01-raw.png")
        junocam_img = JunoCamImage(fname)

        print(f"\n   Product ID: {junocam_img.product_id}")
        print(f"   Image time: {junocam_img.image_time}")

        # Create debug visualization
        print("\n4. Creating debug visualization...")
        out_dir = Path("images/processed")
        out_dir.mkdir(parents=True, exist_ok=True)
        create_debug_visualization(fname, out_dir / "debug_frame_structure.png")

        # Get timing and apply SPICE corrections
        start_et = junocam_img.get_ephemeris_time()
        interframe_delay_str = junocam_img.metadata.get("INTERFRAME_DELAY", "0.371 <s>")
        interframe_delay = float(interframe_delay_str.split()[0])

        # Apply timing corrections from SPICE instrument kernel
        start_time_bias = spice.gdpool("INS-61500_START_TIME_BIAS", 0, 1)[0]
        interframe_delta = spice.gdpool("INS-61500_INTERFRAME_DELTA", 0, 1)[0]

        start_et += start_time_bias
        interframe_delay += interframe_delta

        print(f"   Start ET: {start_et:.2f}")
        print(f"   Interframe delay: {interframe_delay:.3f} seconds")

        # Extract framelets
        print("\n5. Extracting framelets...")
        framelets_by_color = extract_framelets(fname, start_et, interframe_delay)

        # Always use the middle frame
        num_frames = len(framelets_by_color["green"])
        view_frame_idx = num_frames // 2
        reference_framelet = framelets_by_color["green"][view_frame_idx]

        print(f"\n6. Using frame {view_frame_idx} for view reference")
        print(f"   View ET: {reference_framelet.et:.2f}")

        # Create view
        output_rgb, output_red, output_green, output_blue = project_framelets_to_pinhole_view(
            framelets_by_color, ellipsoid, reference_framelet
        )

        if output_rgb is not None:
            # Save all outputs
            output_dir = Path("images/processed/simple")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save RGB composite
            rgb_file = output_dir / f"{junocam_img.product_id}_simple_view_rgb.png"
            cv2.imwrite(str(rgb_file), output_rgb)
            print(f"\n✓ Saved RGB composite: {rgb_file}")

            # Save individual channels
            red_file = output_dir / f"{junocam_img.product_id}_simple_view_red.png"
            cv2.imwrite(str(red_file), output_red)
            print(f"✓ Saved red channel: {red_file}")

            green_file = output_dir / f"{junocam_img.product_id}_simple_view_green.png"
            cv2.imwrite(str(green_file), output_green)
            print(f"✓ Saved green channel: {green_file}")

            blue_file = output_dir / f"{junocam_img.product_id}_simple_view_blue.png"
            cv2.imwrite(str(blue_file), output_blue)
            print(f"✓ Saved blue channel: {blue_file}")
        else:
            print("\n✗ Could not create view - Jupiter not visible in selected frame")

    finally:
        km.unload_kernels()
        print("\n✓ SPICE kernels unloaded")


if __name__ == "__main__":
    main()
