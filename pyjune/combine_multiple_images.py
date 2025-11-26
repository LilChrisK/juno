"""
Combine multiple JunoCam raw images into a single cylindrical projection.

This script processes multiple JunoCam observations and combines them into
a single cylindrical map. Later observations overwrite earlier ones in
overlapping regions.
"""

import cv2
import numpy as np
from pathlib import Path
import spiceypy as spice

from src.spice_correction import SpiceKernelManager, JunoCamImage
from src.map_projection import JupiterEllipsoid
from src.pinhole_projection import extract_framelets
from src.cylindrical_projection import CylindricalProjection
from src.framelet_sampling import CameraParameters


# ============================================================================
# CONFIGURATION - Edit these values as needed
# ============================================================================

# List of raw images to combine (processed in order - later images overwrite earlier ones)
IMAGE_PATHS = [
    "images/raw/JNCE_2021159_34C00048_V01-raw.png",
    "images/raw/JNCE_2021159_34C00055_V01-raw.png",
    "images/raw/JNCE_2021159_34C00080_V01-raw.png",
]

# Cylindrical projection parameters
PROJECTION_CONFIG = {
    "lon_min": 0.0,          # Minimum longitude (degrees West, 0-360)
    "lon_max": 360.0,        # Maximum longitude (degrees West, 0-360)
    "lat_min": -90.0,        # Minimum latitude (degrees, -90 to +90)
    "lat_max": 90.0,         # Maximum latitude (degrees, -90 to +90)
    "resolution_deg": 0.1    # Map resolution in degrees per pixel
}

# Output configuration
OUTPUT_DIR = Path("images/processed/cylindrical")
OUTPUT_NAME = "combined"  # Output files will be named: combined_cylindrical_rgb.png, etc.

# ============================================================================


def process_single_image(
    image_path: Path,
    projection: CylindricalProjection,
    ellipsoid: JupiterEllipsoid,
    camera_params: CameraParameters,
    sun_position: np.ndarray = None
):
    """
    Process a single raw image and add its framelets to the projection.

    Args:
        image_path: Path to raw image file
        projection: CylindricalProjection instance to accumulate data
        ellipsoid: Jupiter ellipsoid model
        camera_params: CameraParameters instance with intrinsic camera parameters
        sun_position: Optional pre-computed sun position (if None, will compute)

    Returns:
        dict with processing statistics
    """
    print("\n" + "-" * 70)
    print(f"Processing: {image_path.name}")
    print("-" * 70)

    # Load image metadata
    if not image_path.exists():
        print(f"✗ Error: Image file not found: {image_path}")
        return None

    junocam_img = JunoCamImage(image_path)
    print(f"Product ID: {junocam_img.product_id}")
    print(f"Image time: {junocam_img.image_time}")

    # Get timing information
    start_et = junocam_img.get_ephemeris_time()
    interframe_delay_str = junocam_img.metadata.get("INTERFRAME_DELAY", "0.371 <s>")
    interframe_delay = float(interframe_delay_str.split()[0])

    # Apply timing corrections from SPICE instrument kernel
    start_time_bias = spice.gdpool("INS-61500_START_TIME_BIAS", 0, 1)[0]
    interframe_delta = spice.gdpool("INS-61500_INTERFRAME_DELTA", 0, 1)[0]

    start_et += start_time_bias
    interframe_delay += interframe_delta

    print(f"Start ET: {start_et:.2f}")
    print(f"Interframe delay: {interframe_delay:.3f} seconds")

    # Extract framelets
    print("Extracting framelets...")
    framelets_by_color = extract_framelets(image_path, start_et, interframe_delay)

    # Use middle frame for reference ephemeris time
    num_frames = len(framelets_by_color["green"])
    view_frame_idx = num_frames // 2
    reference_et = framelets_by_color["green"][view_frame_idx].et

    print(f"Using frame {view_frame_idx}/{num_frames} for reference")
    print(f"Reference ET: {reference_et:.2f}")

    # Get Sun position if not provided
    if sun_position is None:
        sun_position, _ = spice.spkpos('SUN', reference_et, 'IAU_JUPITER', 'LT+S', 'JUPITER')
        print(f"Sun position in IAU_JUPITER frame: {sun_position}")

    # Add framelets to projection
    print("\nAdding framelets to map...")
    total_samples = 0

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
                camera_params,
                sun_position,
                color_name
            )

            total_samples += debug_info['valid']

            if idx % 5 == 0:
                print(f"  Frame {framelet.frame_number:3d}: {debug_info['valid']:,} valid samples")

    print(f"\n✓ Added {total_samples:,} total samples from {image_path.name}")

    return {
        "product_id": junocam_img.product_id,
        "total_samples": total_samples,
        "num_framelets": num_frames
    }


def main():
    """Main entry point for combining multiple images."""
    print("=" * 70)
    print("COMBINE MULTIPLE JUNOCAM IMAGES INTO CYLINDRICAL PROJECTION")
    print("=" * 70)
    print(f"\nImages to process: {len(IMAGE_PATHS)}")
    for i, path in enumerate(IMAGE_PATHS, 1):
        print(f"  {i}. {path}")

    print(f"\nProjection configuration:")
    print(f"  Longitude range: {PROJECTION_CONFIG['lon_min']:.1f}° to {PROJECTION_CONFIG['lon_max']:.1f}° West")
    print(f"  Latitude range: {PROJECTION_CONFIG['lat_min']:.1f}° to {PROJECTION_CONFIG['lat_max']:.1f}°")
    print(f"  Resolution: {PROJECTION_CONFIG['resolution_deg']:.3f} deg/pixel")

    # Load SPICE kernels
    print("\n" + "=" * 70)
    print("1. Loading SPICE kernels...")
    print("=" * 70)
    km = SpiceKernelManager()
    km.load_kernels()

    try:
        # Initialize Jupiter ellipsoid
        print("\n" + "=" * 70)
        print("2. Initializing Jupiter ellipsoid...")
        print("=" * 70)
        ellipsoid = JupiterEllipsoid()

        # Initialize camera parameters
        print("\n" + "=" * 70)
        print("3. Loading camera parameters...")
        print("=" * 70)
        camera_params = CameraParameters()

        # Create cylindrical projection (shared across all images)
        print("\n" + "=" * 70)
        print("4. Creating cylindrical projection...")
        print("=" * 70)
        projection = CylindricalProjection(
            lon_min=PROJECTION_CONFIG['lon_min'],
            lon_max=PROJECTION_CONFIG['lon_max'],
            lat_min=PROJECTION_CONFIG['lat_min'],
            lat_max=PROJECTION_CONFIG['lat_max'],
            resolution_deg=PROJECTION_CONFIG['resolution_deg']
        )

        # Compute surface grid (done once, reused for all images)
        print("\n" + "=" * 70)
        print("5. Computing surface grid...")
        print("=" * 70)
        # Use current time as reference (doesn't matter since IAU_JUPITER is body-fixed)
        reference_et = spice.str2et("2021-06-09T00:00:00")
        projection.compute_surface_grid(ellipsoid, reference_et)

        # Process each image
        print("\n" + "=" * 70)
        print("6. Processing images...")
        print("=" * 70)

        processing_stats = []
        for image_path in IMAGE_PATHS:
            stats = process_single_image(
                Path(image_path),
                projection,
                ellipsoid,
                camera_params
            )
            if stats:
                processing_stats.append(stats)

        # Save combined projection
        print("\n" + "=" * 70)
        print("7. Saving combined projection...")
        print("=" * 70)
        projection.save(OUTPUT_DIR, OUTPUT_NAME)

        # Print summary
        print("\n" + "=" * 70)
        print("✓ PROCESSING COMPLETE")
        print("=" * 70)
        print(f"\nProcessed {len(processing_stats)} images:")
        for stats in processing_stats:
            print(f"  {stats['product_id']}: {stats['total_samples']:,} samples from {stats['num_framelets']} framelets")

        # Get final coverage statistics
        red, green, blue = projection.get_maps()
        total_pixels = projection.width * projection.height
        valid_pixels = np.sum(projection.map_counts > 0)
        coverage_percent = valid_pixels / total_pixels * 100

        print(f"\nFinal map coverage: {coverage_percent:.1f}% ({valid_pixels:,} / {total_pixels:,} pixels)")
        print(f"Output directory: {OUTPUT_DIR}")

    finally:
        km.unload_kernels()
        print("\n✓ SPICE kernels unloaded")


if __name__ == "__main__":
    main()
