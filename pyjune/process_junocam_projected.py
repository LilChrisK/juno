"""
Process JunoCam images with map projection onto Jupiter's ellipsoid.

This script projects pushframe images onto Jupiter's surface using accurate
SPICE-derived geometry and creates orthographic map products.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import json
import time

from spice_correction import SpiceKernelManager, JunoCamImage
from map_projection import JupiterEllipsoid, FrameletProjector, OrthographicProjector
from main import Framelet


def extract_framelets(fname: Path) -> dict:
    """
    Extract framelets organized by color from raw JunoCam image.

    Args:
        fname: Path to raw image file

    Returns:
        Dictionary mapping color names to lists of Framelet objects
    """
    # Load raw image
    raw = cv2.imread(str(fname), cv2.IMREAD_UNCHANGED)
    if raw is None:
        print(f"ERROR: Could not open raw image: {fname}")
        sys.exit(1)

    height, width = raw.shape[:2]
    print(f"Raw image size: {width} x {height}")

    # JunoCam parameters
    band_height = 128  # pixels per color band
    bands = 3  # R, G, B filters
    frame_height = band_height * bands

    # Calculate number of frames
    num_frames = height // frame_height
    print(f"Number of frames: {num_frames}")

    # Color mapping (order in raw image: Blue, Green, Red)
    color_map = {0: "blue", 1: "green", 2: "red"}

    framelets_by_color = {"red": [], "green": [], "blue": []}

    # Extract framelets
    for frame_idx in range(num_frames):
        for color_idx in range(bands):
            base_row = frame_idx * frame_height + color_idx * band_height
            framelet_data = raw[base_row : base_row + band_height, :]

            framelet = Framelet(
                frame_number=frame_idx,
                color=color_map[color_idx],
                color_index=color_idx,
                data=framelet_data.copy(),
            )
            framelets_by_color[framelet.color].append(framelet)

    total_framelets = sum(len(v) for v in framelets_by_color.values())
    print(f"Extracted {total_framelets} framelets:")
    print(f"  Red:   {len(framelets_by_color['red'])}")
    print(f"  Green: {len(framelets_by_color['green'])}")
    print(f"  Blue:  {len(framelets_by_color['blue'])}")

    return framelets_by_color


def compute_framelet_timing(
    start_et: float,
    interframe_delay: float,
    frame_number: int,
    color: str
) -> float:
    """
    Compute ephemeris time for a specific framelet.

    Args:
        start_et: Start ephemeris time of first frame
        interframe_delay: Time between frames (seconds)
        frame_number: Frame index
        color: Color name ('blue', 'green', 'red')

    Returns:
        Ephemeris time for this framelet
    """
    # Base time for this frame
    frame_et = start_et + frame_number * interframe_delay

    # Color-specific timing offset within frame
    # Order of acquisition: Blue, Green, Red
    # Approximate timing between color filters (milliseconds)
    filter_delay = 0.001  # 1 ms between filters (approximate)

    color_offsets = {
        'blue': 0.0,  # First exposure
        'green': filter_delay,  # Second exposure
        'red': 2.0 * filter_delay,  # Third exposure
    }

    return frame_et + color_offsets[color]


def determine_map_center(junocam_img: JunoCamImage, ellipsoid: JupiterEllipsoid) -> tuple:
    """
    Determine center point for orthographic projection.

    Uses the boresight of the middle frame to find a natural center.

    Args:
        junocam_img: Image metadata
        ellipsoid: Jupiter ellipsoid

    Returns:
        (center_lon, center_lat) in degrees
    """
    import spiceypy as spice

    # Get middle frame time
    et = junocam_img.get_ephemeris_time()

    if junocam_img.metadata:
        interframe_delay_str = junocam_img.metadata.get("INTERFRAME_DELAY", "0.371 <s>")
        interframe_delay = float(interframe_delay_str.split()[0])

        # Get rough frame count
        # Estimate: use stop time
        start_time = junocam_img.metadata.get("START_TIME", "")
        stop_time = junocam_img.metadata.get("STOP_TIME", "")

        if start_time and stop_time:
            et_start = spice.str2et(start_time)
            et_stop = spice.str2et(stop_time)
            num_frames = int((et_stop - et_start) / interframe_delay)
            et_middle = et_start + (num_frames / 2) * interframe_delay
        else:
            et_middle = et
    else:
        et_middle = et

    # Get spacecraft position
    state, _ = spice.spkezr('JUNO', et_middle, 'J2000', 'NONE', 'JUPITER')
    sc_position = state[:3]

    # Get camera boresight
    try:
        rotation = spice.pxform('JUNO_JUNOCAM', 'J2000', et_middle)
        shape, frame_name, boresight_inst, n, fov_bounds = spice.getfov(-61500, 16)
        boresight_j2000 = rotation @ boresight_inst

        # Intersect boresight with Jupiter
        intersection = ellipsoid.ray_intersection(sc_position, boresight_j2000)

        if intersection is not None:
            lon, lat, alt = ellipsoid.cartesian_to_planetographic(intersection, et_middle)
            print(f"\nMap center from boresight intersection:")
            print(f"  Longitude: {lon:.2f}° W")
            print(f"  Latitude:  {lat:.2f}°")
            return lon, lat

    except Exception as e:
        print(f"Could not determine boresight: {e}")

    # Fallback: sub-spacecraft point
    # Direction from Jupiter to spacecraft
    direction = -sc_position / np.linalg.norm(sc_position)
    intersection = ellipsoid.ray_intersection(np.array([0, 0, 0]), direction)

    if intersection is not None:
        lon, lat, alt = ellipsoid.cartesian_to_planetographic(intersection, et_middle)
        print(f"\nMap center from sub-spacecraft point:")
        print(f"  Longitude: {lon:.2f}° W")
        print(f"  Latitude:  {lat:.2f}°")
        return lon, lat

    # Ultimate fallback
    print("\nUsing default map center: (0°, 0°)")
    return 0.0, 0.0


def project_junocam_to_map(
    fname: Path,
    junocam_img: JunoCamImage,
    ellipsoid: JupiterEllipsoid,
    output_dir: Path,
    map_size: int = 2048,
    sample_step: int = 4
):
    """
    Project JunoCam image onto orthographic map.

    Args:
        fname: Path to raw image
        junocam_img: Image metadata
        ellipsoid: Jupiter ellipsoid model
        output_dir: Output directory for maps
        map_size: Output map size in pixels
        sample_step: Sample every Nth pixel from framelets
    """
    print("\n" + "=" * 70)
    print("MAP PROJECTION")
    print("=" * 70)

    # Extract framelets
    print("\n1. Extracting framelets...")
    framelets_by_color = extract_framelets(fname)

    # Get timing information
    print("\n2. Computing timing...")
    start_et = junocam_img.get_ephemeris_time()

    if junocam_img.metadata:
        interframe_delay_str = junocam_img.metadata.get("INTERFRAME_DELAY", "0.371 <s>")
        interframe_delay = float(interframe_delay_str.split()[0])
        print(f"   Interframe delay: {interframe_delay:.3f} seconds")
    else:
        print("   WARNING: No metadata, using default 0.371s")
        interframe_delay = 0.371

    # Determine map center
    print("\n3. Determining map center...")
    center_lon, center_lat = determine_map_center(junocam_img, ellipsoid)

    # Estimate scale based on range
    import spiceypy as spice
    state, _ = spice.spkezr('JUNO', start_et, 'J2000', 'NONE', 'JUPITER')
    range_km = np.linalg.norm(state[:3])

    # Scale to show ~1.5x Jupiter diameter in frame
    scale_km_per_pixel = (ellipsoid.a * 3.0) / map_size
    print(f"   Map scale: {scale_km_per_pixel:.2f} km/pixel")
    print(f"   Spacecraft range: {range_km:,.0f} km")

    # Create orthographic projector
    print("\n4. Creating orthographic projector...")
    projector = OrthographicProjector(
        ellipsoid=ellipsoid,
        center_lon=center_lon,
        center_lat=center_lat,
        et=start_et,
        map_width=map_size,
        map_height=map_size,
        scale_km_per_pixel=scale_km_per_pixel
    )

    # Project each color channel
    print("\n5. Projecting framelets...")

    # TEMPORARY: Only process green channel for testing
    for color in ['green']:
        framelets = framelets_by_color[color]
        print(f"\n   Processing {color.upper()} channel ({len(framelets)} framelets)...")

        # Timing benchmarks
        time_init = 0.0
        time_project = 0.0
        time_accumulate = 0.0
        total_points = 0

        for idx, framelet in enumerate(framelets):
            # Skip incomplete first and last frames
            if framelet.frame_number == 0 or framelet.frame_number >= len(framelets) - 1:
                continue

            frame_start = time.time()

            # Compute timing
            et = compute_framelet_timing(
                start_et,
                interframe_delay,
                framelet.frame_number,
                color
            )

            # Create framelet projector
            try:
                t0 = time.time()
                fp = FrameletProjector(
                    ellipsoid=ellipsoid,
                    et=et,
                    framelet_index=framelet.frame_number,
                    color=color
                )
                t1 = time.time()
                time_init += (t1 - t0)

                # Project framelet pixels to surface
                t0 = time.time()
                surface_points = fp.project_framelet(
                    framelet.data,
                    sample_step=sample_step
                )
                t1 = time.time()
                time_project += (t1 - t0)

                # Add points to map
                t0 = time.time()
                for point in surface_points:
                    # Get pixel value from framelet
                    pixel_value = framelet.data[point.pixel_y, point.pixel_x]

                    # Add to map
                    projector.add_surface_point(point, float(pixel_value), color)
                t1 = time.time()
                time_accumulate += (t1 - t0)

                total_points += len(surface_points)

                frame_elapsed = time.time() - frame_start
                if idx % 5 == 0:
                    print(f"      Frame {framelet.frame_number:3d}: {len(surface_points):5d} points, {frame_elapsed:.2f}s")

            except Exception as e:
                print(f"      Frame {framelet.frame_number:3d}: ERROR - {e}")
                continue

        # Print timing summary
        print(f"\n   Timing breakdown:")
        print(f"     Initialization: {time_init:.2f}s")
        print(f"     Projection:     {time_project:.2f}s")
        print(f"     Accumulation:   {time_accumulate:.2f}s")
        print(f"     Total points:   {total_points:,}")

    # Get final maps
    print("\n6. Generating final maps...")
    map_red, map_green, map_blue = projector.get_maps()

    # Save individual channels
    print("\n7. Saving output files...")
    output_dir.mkdir(parents=True, exist_ok=True)

    product_id = junocam_img.product_id

    # Save as 16-bit grayscale
    def save_channel(data, filename):
        # Normalize to 16-bit range
        if data.max() > 0:
            normalized = (data / data.max() * 65535).astype(np.uint16)
        else:
            normalized = np.zeros_like(data, dtype=np.uint16)
        cv2.imwrite(str(output_dir / filename), normalized)
        print(f"   Saved: {filename}")

    # TEMPORARY: Only save green channel for testing
    save_channel(map_green, f"{product_id}_map_green.png")

    # Also save as 8-bit for easier viewing
    if map_green.max() > 0:
        green8 = (map_green / map_green.max() * 255).astype(np.uint8)
    else:
        green8 = np.zeros_like(map_green, dtype=np.uint8)
    cv2.imwrite(str(output_dir / f"{product_id}_map_green_8bit.png"), green8)
    print(f"   Saved: {product_id}_map_green_8bit.png")

    # Save metadata
    metadata = {
        "product_id": product_id,
        "projection": "orthographic",
        "center_longitude_deg": center_lon,
        "center_latitude_deg": center_lat,
        "map_width_pixels": map_size,
        "map_height_pixels": map_size,
        "scale_km_per_pixel": scale_km_per_pixel,
        "ellipsoid_radii_km": {
            "equatorial_a": ellipsoid.a,
            "equatorial_b": ellipsoid.b,
            "polar_c": ellipsoid.c
        },
        "sample_step": sample_step,
        "spacecraft_range_km": float(range_km)
    }

    with open(output_dir / f"{product_id}_map_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Saved: {product_id}_map_metadata.json")

    print("\n" + "=" * 70)
    print("MAP PROJECTION COMPLETE!")
    print("=" * 70)


def main():
    print("=" * 70)
    print("JUNOCAM MAP PROJECTION")
    print("=" * 70)

    # Load SPICE kernels
    print("\n1. Loading SPICE kernels...")
    km = SpiceKernelManager()
    km.load_kernels()

    try:
        # Initialize Jupiter ellipsoid from SPICE
        print("\n2. Initializing Jupiter ellipsoid...")
        ellipsoid = JupiterEllipsoid()

        # Load image metadata
        print("\n3. Loading image metadata...")
        fname = Path("images/raw/JNCE_2021159_34C00080_V01-raw.png")
        # fname = Path("images/raw/JNCE_2021159_34C00055_V01-raw.png")
        # fname = Path("images/raw/JNCE_2021159_34C00048_V01-raw.png")

        junocam_img = JunoCamImage(fname)

        print(f"\n   Product ID: {junocam_img.product_id}")
        print(f"   Date: {junocam_img.year}-{junocam_img.doy:03d}")
        print(f"   Orbit: {junocam_img.orbit}")

        if junocam_img.metadata:
            print(f"   Image time: {junocam_img.image_time}")
        else:
            print("   WARNING: No metadata found!")

        # Project to map
        output_dir = Path("images/processed/maps")
        project_junocam_to_map(
            fname=fname,
            junocam_img=junocam_img,
            ellipsoid=ellipsoid,
            output_dir=output_dir,
            map_size=2048,
            sample_step=1  # Dense sampling for single channel testing
        )

    finally:
        # Unload SPICE kernels
        km.unload_kernels()
        print("\n✓ SPICE kernels unloaded")


if __name__ == "__main__":
    main()
