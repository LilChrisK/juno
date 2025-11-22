"""
JunoCam image processing with SPICE kernel support.
Main entry point for processing JunoCam images.

Supports both pinhole camera views and cylindrical map projections.
"""

import cv2
import argparse
from pathlib import Path
import spiceypy as spice

from spice_correction import SpiceKernelManager, JunoCamImage
from map_projection import JupiterEllipsoid
from utils import create_debug_visualization
from pinhole_projection import extract_framelets, project_framelets_to_pinhole_view
from cylindrical_projection import CylindricalProjection


def process_pinhole(framelets_by_color, ellipsoid, reference_framelet, junocam_img):
    """Process framelets to create pinhole camera view."""
    print("\n" + "=" * 70)
    print("CREATING PINHOLE VIEW")
    print("=" * 70)

    output_rgb, output_red, output_green, output_blue = project_framelets_to_pinhole_view(
        framelets_by_color, ellipsoid, reference_framelet
    )

    if output_rgb is not None:
        # Save all outputs
        output_dir = Path("images/processed/pinhole")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save RGB composite
        rgb_file = output_dir / f"{junocam_img.product_id}_pinhole_rgb.png"
        cv2.imwrite(str(rgb_file), output_rgb)
        print(f"\n✓ Saved RGB composite: {rgb_file}")

        # Save individual channels
        red_file = output_dir / f"{junocam_img.product_id}_pinhole_red.png"
        cv2.imwrite(str(red_file), output_red)
        print(f"✓ Saved red channel: {red_file}")

        green_file = output_dir / f"{junocam_img.product_id}_pinhole_green.png"
        cv2.imwrite(str(green_file), output_green)
        print(f"✓ Saved green channel: {green_file}")

        blue_file = output_dir / f"{junocam_img.product_id}_pinhole_blue.png"
        cv2.imwrite(str(blue_file), output_blue)
        print(f"✓ Saved blue channel: {blue_file}")
    else:
        print("\n✗ Could not create view - Jupiter not visible in selected frame")


def process_cylindrical(
    framelets_by_color,
    ellipsoid,
    reference_et,
    junocam_img,
    lon_min=0.0,
    lon_max=360.0,
    lat_min=-90.0,
    lat_max=90.0,
    resolution_deg=0.1
):
    """Process framelets to create cylindrical map projection."""
    print("\n" + "=" * 70)
    print("CREATING CYLINDRICAL PROJECTION")
    print("=" * 70)

    # Create projection
    projection = CylindricalProjection(
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        resolution_deg=resolution_deg
    )

    # Compute surface grid
    print("\nComputing surface grid...")
    projection.compute_surface_grid(ellipsoid, reference_et)

    # Get Sun position (single query for entire image)
    sun_position, _ = spice.spkpos('SUN', reference_et, 'IAU_JUPITER', 'LT+S', 'JUPITER')
    print(f"\nSun position in IAU_JUPITER frame: {sun_position}")

    # Add framelets
    print("\nProcessing framelets...")

    for color_name in ['red', 'green', 'blue']:
        framelets = framelets_by_color[color_name]
        print(f"\n{color_name.upper()} channel ({len(framelets)} framelets):")

        for idx, framelet in enumerate(framelets):
            # Skip first and last frames (often incomplete)
            if framelet.frame_number == 0 or framelet.frame_number >= len(framelets) - 1:
                continue

            debug_info = projection.add_framelet(
                framelet.data,
                framelet.cam_position,
                framelet.cam_orient,
                ellipsoid,
                sun_position,
                color_name
            )

            if idx % 5 == 0:
                print(f"  Frame {framelet.frame_number:3d}: {debug_info['valid']:,} valid samples")

    # Save projection
    output_dir = Path("images/processed/cylindrical")
    projection.save(output_dir, junocam_img.product_id)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Process JunoCam images to create pinhole views or cylindrical map projections.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create pinhole view (default)
  python main.py

  # Create cylindrical projection
  python main.py --mode cylindrical

  # Create both
  python main.py --mode both

  # Custom cylindrical bounds and resolution
  python main.py --mode cylindrical --cyl-resolution 0.05 --cyl-lat-range -60 60

  # Specify input image
  python main.py --image images/raw/JNCE_2021159_34C00055_V01-raw.png --mode cylindrical
        """
    )

    parser.add_argument(
        '--mode',
        choices=['pinhole', 'cylindrical', 'both'],
        default='pinhole',
        help='Processing mode: pinhole view, cylindrical projection, or both (default: pinhole)'
    )

    parser.add_argument(
        '--image',
        type=str,
        default='images/raw/JNCE_2021159_34C00080_V01-raw.png',
        # default='images/raw/JNCE_2021159_34C00055_V01-raw.png',
        # default='images/raw/JNCE_2021159_34C00048_V01-raw.png',
        help='Path to input image (default: JNCE_2021159_34C00080_V01-raw.png)'
    )

    parser.add_argument(
        '--cyl-resolution',
        type=float,
        default=0.1,
        help='Cylindrical projection resolution in degrees/pixel (default: 0.1)'
    )

    parser.add_argument(
        '--cyl-lon-range',
        type=float,
        nargs=2,
        default=[0.0, 360.0],
        metavar=('MIN', 'MAX'),
        help='Longitude range for cylindrical projection in degrees West (default: 0 360)'
    )

    parser.add_argument(
        '--cyl-lat-range',
        type=float,
        nargs=2,
        default=[-90.0, 90.0],
        metavar=('MIN', 'MAX'),
        help='Latitude range for cylindrical projection in degrees (default: -90 90)'
    )

    args = parser.parse_args()

    # Print header
    print("=" * 70)
    print("JUNOCAM IMAGE PROCESSING")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Image: {args.image}")

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
        fname = Path(args.image)
        if not fname.exists():
            print(f"\n✗ Error: Image file not found: {fname}")
            return

        junocam_img = JunoCamImage(fname)

        print(f"\n   Product ID: {junocam_img.product_id}")
        print(f"   Image time: {junocam_img.image_time}")

        # Get timing and apply SPICE corrections
        print("\n4. Getting timing information...")
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

        # Use middle frame for reference
        num_frames = len(framelets_by_color["green"])
        view_frame_idx = num_frames // 2
        reference_framelet = framelets_by_color["green"][view_frame_idx]

        print(f"\n6. Using frame {view_frame_idx}/{num_frames} for reference")
        print(f"   Reference ET: {reference_framelet.et:.2f}")

        # Process based on mode
        if args.mode in ['pinhole', 'both']:
            process_pinhole(framelets_by_color, ellipsoid, reference_framelet, junocam_img)

        if args.mode in ['cylindrical', 'both']:
            process_cylindrical(
                framelets_by_color,
                ellipsoid,
                reference_framelet.et,
                junocam_img,
                lon_min=args.cyl_lon_range[0],
                lon_max=args.cyl_lon_range[1],
                lat_min=args.cyl_lat_range[0],
                lat_max=args.cyl_lat_range[1],
                resolution_deg=args.cyl_resolution
            )

        print("\n" + "=" * 70)
        print("✓ PROCESSING COMPLETE")
        print("=" * 70)

    finally:
        km.unload_kernels()
        print("\n✓ SPICE kernels unloaded")


if __name__ == "__main__":
    main()
