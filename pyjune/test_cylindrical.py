"""
Test cylindrical projection with actual JunoCam data.
"""

from pathlib import Path
import spiceypy as spice

from spice_correction import SpiceKernelManager, JunoCamImage
from map_projection import JupiterEllipsoid
from pinhole_projection import extract_framelets
from cylindrical_projection import CylindricalProjection


def main():
    print("=" * 70)
    print("CYLINDRICAL PROJECTION TEST")
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
        junocam_img = JunoCamImage(fname)

        # Get timing
        start_et = junocam_img.get_ephemeris_time()
        interframe_delay_str = junocam_img.metadata.get("INTERFRAME_DELAY", "0.371 <s>")
        interframe_delay = float(interframe_delay_str.split()[0])

        # Apply SPICE corrections
        start_time_bias = spice.gdpool("INS-61500_START_TIME_BIAS", 0, 1)[0]
        interframe_delta = spice.gdpool("INS-61500_INTERFRAME_DELTA", 0, 1)[0]

        start_et += start_time_bias
        interframe_delay += interframe_delta

        print(f"   Start ET: {start_et:.2f}")

        # Extract framelets
        print("\n4. Extracting framelets...")
        framelets_by_color = extract_framelets(fname, start_et, interframe_delay)

        # Use middle frame for reference time
        num_frames = len(framelets_by_color["green"])
        reference_frame_idx = num_frames // 2
        reference_et = framelets_by_color["green"][reference_frame_idx].et

        print(f"   Using frame {reference_frame_idx}/{num_frames} as reference")
        print(f"   Reference ET: {reference_et:.2f}")

        # Create cylindrical projection
        # Test with a smaller region first (faster)
        print("\n5. Creating cylindrical projection...")
        print("   Using reduced region for testing (0-360° lon, -60 to +60° lat)")

        projection = CylindricalProjection(
            lon_min=0.0,
            lon_max=360.0,
            lat_min=-60.0,
            lat_max=60.0,
            resolution_deg=0.1  # 0.1 degrees per pixel
        )

        # Compute surface grid
        print("\n6. Computing surface grid...")
        projection.compute_surface_grid(ellipsoid, reference_et)

        # Add framelets
        print("\n7. Processing framelets...")

        for color_name in ['red', 'green', 'blue']:
            framelets = framelets_by_color[color_name]
            print(f"\n   {color_name.upper()} channel ({len(framelets)} framelets):")

            for idx, framelet in enumerate(framelets):
                # Skip first and last frames (often incomplete)
                if framelet.frame_number == 0 or framelet.frame_number >= len(framelets) - 1:
                    continue

                debug_info = projection.add_framelet(
                    framelet.data,
                    framelet.cam_position,
                    framelet.cam_orient,
                    color_name
                )

                if idx % 5 == 0:
                    print(f"      Frame {framelet.frame_number:3d}: {debug_info['valid']:,} valid samples")

        # Save projection
        print("\n8. Saving projection...")
        output_dir = Path("images/processed/cylindrical")
        projection.save(output_dir, junocam_img.product_id)

        # Test loading
        print("\n9. Testing load functionality...")
        metadata_path = output_dir / f"{junocam_img.product_id}_cylindrical_metadata.json"
        loaded_projection = CylindricalProjection.load(metadata_path)

        # Test coordinate conversion
        print("\n10. Testing coordinate conversions...")
        test_coords = [
            (0.0, 0.0, "Equator, Prime Meridian"),
            (45.0, 180.0, "Mid-latitude, Antimeridian"),
            (-30.0, 90.0, "Southern mid-latitude")
        ]

        for lat, lon, desc in test_coords:
            px, py = projection.latlon_to_pixel(lat, lon)
            lat_back, lon_back = projection.pixel_to_latlon(px, py)
            error_lat = abs(lat_back - lat)
            error_lon = abs(lon_back - lon)

            print(f"   {desc}:")
            print(f"      Input: ({lat:.2f}°, {lon:.2f}°) → Pixel: ({px:.1f}, {py:.1f})")
            print(f"      Back: ({lat_back:.2f}°, {lon_back:.2f}°)")
            print(f"      Error: ({error_lat:.6f}°, {error_lon:.6f}°)")

        print("\n" + "=" * 70)
        print("✓ CYLINDRICAL PROJECTION TEST COMPLETE")
        print("=" * 70)

    finally:
        km.unload_kernels()


if __name__ == "__main__":
    main()
