"""
Simple JunoCam processing - replicates Example1.py approach.
Creates a synthetic pinhole camera view by sampling framelets.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import spiceypy as spice
from scipy.interpolate import RectBivariateSpline

from spice_correction import SpiceKernelManager, JunoCamImage
from map_projection import JupiterEllipsoid, get_junocam_fov
from main import Framelet


def extract_framelets(fname: Path, start_et: float = 0.0, interframe_delay: float = 0.0) -> dict:
    """Extract framelets organized by color from raw JunoCam image."""
    raw = cv2.imread(str(fname), cv2.IMREAD_UNCHANGED)
    if raw is None:
        print(f"ERROR: Could not open raw image: {fname}")
        sys.exit(1)

    height, width = raw.shape[:2]
    print(f"Raw image size: {width} x {height}")

    band_height = 128
    bands = 3
    frame_height = band_height * bands
    num_frames = height // frame_height
    print(f"Number of frames: {num_frames}")

    color_map = {0: "blue", 1: "green", 2: "red"}
    framelets_by_color = {"red": [], "green": [], "blue": []}

    for frame_idx in range(num_frames):
        frame_et = start_et + frame_idx * interframe_delay

        for color_idx in range(bands):
            base_row = frame_idx * frame_height + color_idx * band_height
            framelet_data = raw[base_row : base_row + band_height, :]

            framelet = Framelet(
                frame_number=frame_idx,
                color=color_map[color_idx],
                color_index=color_idx,
                data=framelet_data.copy(),
                et=frame_et,
            )
            framelets_by_color[framelet.color].append(framelet)

    return framelets_by_color


def create_view_from_framelets(
    framelets_by_color: dict,
    ellipsoid: JupiterEllipsoid,
    view_et: float
):
    """
    Create a synthetic camera view by sampling framelets.
    Automatically sizes the view to capture all of Jupiter.

    Args:
        framelets_by_color: Dictionary of framelets by color
        ellipsoid: Jupiter ellipsoid
        view_et: Ephemeris time for view (typically middle frame)
    """
    print("\n" + "=" * 70)
    print("CREATING SYNTHETIC VIEW")
    print("=" * 70)

    # Get camera state at view time
    print(f"\n1. Getting camera state at ET={view_et:.2f}...")
    state, _ = spice.spkezr('JUNO', view_et, 'J2000', 'NONE', 'JUPITER')
    cam_position = state[:3]
    cam_orient = spice.pxform('JUNO_JUNOCAM', 'J2000', view_et)

    print(f"   Camera position: {cam_position}")
    print(f"   Distance to Jupiter: {np.linalg.norm(cam_position):,.0f} km")

    # Get focal length #TODO: Source? Get from spice? 
    focal_length_pixels = 10.95637 / 0.0074
    print(f"   Focal length: {focal_length_pixels:.1f} pixels")

    fov_data = get_junocam_fov()  # Still needed for sampling function signature

    # Find where Jupiter actually appears by testing sample rays
    print("   Finding Jupiter location in detector...")

    hit_pixels = []
    # Sample VERY densely to get accurate extent
    for x in range(-500, 2148, 10):  # Every 10 pixels, wide range
        for y in range(-500, 628, 10):  # Every 10 pixels, wide range
            ray_inst = np.array([float(x), float(y), focal_length_pixels])
            ray_j2000 = cam_orient @ ray_inst
            ray_j2000 = ray_j2000 / np.linalg.norm(ray_j2000)

            intersection = ellipsoid.ray_intersection(cam_position, ray_j2000)
            if intersection is not None:
                hit_pixels.append((x, y))

    if len(hit_pixels) == 0:
        print("   ERROR: Jupiter not visible in this frame!")
        return None

    hit_pixels = np.array(hit_pixels)
    x_min, x_max = hit_pixels[:, 0].min(), hit_pixels[:, 0].max()
    y_min, y_max = hit_pixels[:, 1].min(), hit_pixels[:, 1].max()

    # Center on the geometric center of the bounding box, not the mean of samples
    x_offset = int((x_min + x_max) / 2)
    y_offset = int((y_min + y_max) / 2)

    # Calculate required view size to capture all of Jupiter with margin
    x_span = x_max - x_min
    y_span = y_max - y_min

    # Use the larger span and make it square with HUGE margin to ensure we get everything
    max_span = max(x_span, y_span)
    view_size = int(max_span * 2)  

    # Clamp to reasonable size, prefer larger
    view_size = max(1024, min(view_size, 2048))

    print(f"   Jupiter center at pixel: ({x_offset}, {y_offset})")
    print(f"   Jupiter range: X=[{x_min:.0f}, {x_max:.0f}], Y=[{y_min:.0f}, {y_max:.0f}]")
    print(f"   Jupiter span: {x_span:.0f} x {y_span:.0f} pixels")
    print(f"   View size: {view_size}x{view_size} pixels (4x margin, min 1024)")

    # Create square view grid
    print(f"\n2. Creating {view_size}x{view_size} view grid...")
    half_size = view_size // 2
    y, x = np.mgrid[-half_size:half_size, -half_size:half_size]
    x = x + x_offset
    y = y + y_offset

    # Create rays in camera pixel coordinates
    rays_camera = np.stack([
        x.astype(np.float32),
        y.astype(np.float32),
        np.full((view_size, view_size), focal_length_pixels, dtype=np.float32)
    ], axis=-1)

    # Transform rays to J2000
    # cam_orient transforms column vectors: v_j2000 = cam_orient @ v_inst
    # For row vectors, we need to transpose: v_j2000 = v_inst @ cam_orient.T
    rays_j2000 = rays_camera @ cam_orient.T

    # Project rays onto Jupiter surface
    print("\n3. Projecting rays onto Jupiter surface...")
    surface_positions = np.full((view_size, view_size, 3), np.nan, dtype=np.float32)

    # Debug: test center pixel
    center_idx = view_size // 2
    center_ray = rays_j2000[center_idx, center_idx]
    print(f"   Testing center pixel ({center_idx}, {center_idx}) -> detector pixel ({x[center_idx, center_idx]:.0f}, {y[center_idx, center_idx]:.0f})")
    print(f"   Expected detector pixel for Jupiter center: ({x_offset}, {y_offset})")
    center_hit = ellipsoid.ray_intersection(cam_position, center_ray)
    print(f"   Center hit Jupiter: {center_hit is not None}")

    hit_count = 0
    for i in range(view_size):
        if i % 100 == 0:
            print(f"   Row {i}/{view_size}... ({hit_count} hits so far)")
        for j in range(view_size):
            ray_dir = rays_j2000[i, j]
            intersection = ellipsoid.ray_intersection(cam_position, ray_dir)
            if intersection is not None:
                surface_positions[i, j] = intersection
                hit_count += 1

    valid_surface = ~np.isnan(surface_positions[..., 0])
    print(f"   Valid surface points: {np.sum(valid_surface):,} / {view_size*view_size:,} ({100*np.sum(valid_surface)/(view_size*view_size):.1f}%)")

    # Sample framelets at surface positions
    print("\n4. Sampling framelets...")

    colors = np.zeros((view_size, view_size, 3), dtype=np.float32)
    color_counts = np.zeros((view_size, view_size, 3), dtype=np.float32)

    # Process green channel only for testing
    for color_name, color_idx in [('green', 1)]:
        framelets = framelets_by_color[color_name]
        print(f"\n   Processing {color_name.upper()} channel ({len(framelets)} framelets)...")

        for idx, framelet in enumerate(framelets):
            if framelet.frame_number == 0 or framelet.frame_number >= len(framelets) - 1:
                continue

            # Get camera state for this framelet
            state, _ = spice.spkezr('JUNO', framelet.et, 'J2000', 'NONE', 'JUPITER')
            fl_cam_pos = state[:3]
            fl_cam_orient = spice.pxform('JUNO_JUNOCAM', 'J2000', framelet.et)

            # Sample framelet at surface positions
            sampled, valid_mask, debug_info = sample_framelet_at_positions(
                surface_positions,
                framelet.data,
                fl_cam_pos,
                fl_cam_orient,
                fov_data,
                ellipsoid
            )

            colors[..., 2-color_idx] += sampled  # BGR order
            color_counts[..., 2-color_idx] += valid_mask

            if idx % 5 == 0:
                print(f"      Frame {framelet.frame_number:3d}: {np.sum(valid_mask):,} valid samples")
                print(f"        Validation: {debug_info['valid']}/{debug_info['total']} valid "
                      f"({debug_info['in_front']}/{debug_info['total']} in front, "
                      f"{debug_info['in_x']}/{debug_info['total']} in X, "
                      f"{debug_info['in_y']}/{debug_info['total']} in Y)")
                print(f"        Pixel ranges: X=[{debug_info['pixel_x_range'][0]:.1f}, {debug_info['pixel_x_range'][1]:.1f}] (valid: 0-{debug_info['framelet_size'][1]-1}), "
                      f"Y=[{debug_info['pixel_y_range'][0]:.1f}, {debug_info['pixel_y_range'][1]:.1f}] (valid: 0-{debug_info['framelet_size'][0]-1})")

    # Check if we got any valid samples
    if color_counts.max() == 0:
        print("\n✗ No valid samples from any framelet!")
        return None

    # Average and normalize
    print("\n5. Generating final image...")
    for c in range(3):
        mask = color_counts[..., c] > 0
        colors[..., c][mask] /= color_counts[..., c][mask]

    # Normalize to 0-255
    if colors.max() > 0:
        colors = colors / colors.max() * 255

    output_image = colors.astype(np.uint8)

    return output_image


def sample_framelet_at_positions(
    surface_positions: np.ndarray,
    framelet_data: np.ndarray,
    cam_pos: np.ndarray,
    cam_orient: np.ndarray,
    fov_data: dict,
    ellipsoid: JupiterEllipsoid
) -> tuple:
    """
    Sample framelet at given surface positions.

    Returns:
        (pixel_values, valid_mask)
    """
    height, width = framelet_data.shape
    output_shape = surface_positions.shape[:-1]

    # Flatten
    flat_pos = surface_positions.reshape(-1, 3)
    valid_surface = ~np.isnan(flat_pos[:, 0])

    # Debug: check input
    # print(f"      [sample_framelet] Input shape: {surface_positions.shape}, valid: {np.sum(valid_surface)}/{len(flat_pos)}")

    pixel_values = np.zeros(len(flat_pos), dtype=np.float32)
    valid_mask = np.zeros(len(flat_pos), dtype=np.float32)

    if not np.any(valid_surface):
        debug_info = {'total': 0, 'in_front': 0, 'in_x': 0, 'in_y': 0, 'valid': 0,
                      'pixel_x_range': (0, 0), 'pixel_y_range': (0, 0),
                      'framelet_size': (height, width)}
        return pixel_values.reshape(output_shape), valid_mask.reshape(output_shape), debug_info

    # Get valid positions
    valid_pos = flat_pos[valid_surface]

    # Transform to camera frame
    rays = valid_pos - cam_pos
    # cam_orient transforms instrument->J2000: v_j2000 = cam_orient @ v_inst
    # So inverse is: v_inst = cam_orient.T @ v_j2000 (for column vectors)
    # For row vectors: v_inst = v_j2000 @ cam_orient (no transpose needed!)
    rays_inst = rays @ cam_orient

    # Use pinhole camera model (like Example1.py)
    # Camera intrinsics from Util.py:
    # fl = 10.95637/0.0074 ≈ 1480.86 pixels
    # cx = 814.21, cy = 3.48 (for green band)
    focal_length = 10.95637 / 0.0074
    cx = 814.21
    cy = 3.48  # Green band

    # Pinhole projection: pixel = (X/Z * f, Y/Z * f) + principal_point
    with np.errstate(divide='ignore', invalid='ignore'):
        pixel_x = (rays_inst[:, 0] / rays_inst[:, 2]) * focal_length + cx
        pixel_y = (rays_inst[:, 1] / rays_inst[:, 2]) * focal_length + cy

    # Debug info
    in_front = rays_inst[:, 2] > 0
    in_x_bounds = (pixel_x >= 0) & (pixel_x < width - 1)
    in_y_bounds = (pixel_y >= 0) & (pixel_y < height - 1)

    # Check valid pixels
    framelet_valid = in_x_bounds & in_y_bounds & in_front

    # Debug - return info about why validation failed
    debug_info = {
        'total': len(valid_pos),
        'in_front': np.sum(in_front),
        'in_x': np.sum(in_x_bounds),
        'in_y': np.sum(in_y_bounds),
        'valid': np.sum(framelet_valid),
        'pixel_x_range': (float(pixel_x.min()), float(pixel_x.max())) if len(pixel_x) > 0 else (0, 0),
        'pixel_y_range': (float(pixel_y.min()), float(pixel_y.max())) if len(pixel_y) > 0 else (0, 0),
        'framelet_size': (height, width),
    }

    if np.any(framelet_valid):
        # Interpolate
        interp_func = RectBivariateSpline(
            np.arange(height),
            np.arange(width),
            framelet_data,
            kx=1, ky=1
        )

        valid_px = pixel_x[framelet_valid]
        valid_py = pixel_y[framelet_valid]

        sampled = interp_func(valid_py, valid_px, grid=False)

        # Map back to full array
        temp_values = np.zeros(len(valid_pos), dtype=np.float32)
        temp_valid = np.zeros(len(valid_pos), dtype=np.float32)
        temp_values[framelet_valid] = sampled
        temp_valid[framelet_valid] = 1.0

        pixel_values[valid_surface] = temp_values
        valid_mask[valid_surface] = temp_valid

    return pixel_values.reshape(output_shape), valid_mask.reshape(output_shape), debug_info


def main():
    print("=" * 70)
    print("JUNOCAM SIMPLE VIEW (Example1.py style)")
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

        print(f"\n   Product ID: {junocam_img.product_id}")
        print(f"   Image time: {junocam_img.image_time}")

        # Get timing
        start_et = junocam_img.get_ephemeris_time()
        interframe_delay_str = junocam_img.metadata.get("INTERFRAME_DELAY", "0.371 <s>")
        interframe_delay = float(interframe_delay_str.split()[0])

        print(f"   Start ET: {start_et:.2f}")
        print(f"   Interframe delay: {interframe_delay:.3f} seconds")

        # Extract framelets
        print("\n4. Extracting framelets...")
        framelets_by_color = extract_framelets(fname, start_et, interframe_delay)

        # Always use the middle frame
        num_frames = len(framelets_by_color['green'])
        view_frame_idx = num_frames // 2
        view_et = start_et + view_frame_idx * interframe_delay

        print(f"\n5. Using frame {view_frame_idx} for view reference")
        print(f"   View ET: {view_et:.2f}")

        # Create view
        output_image = create_view_from_framelets(
            framelets_by_color,
            ellipsoid,
            view_et
        )

        if output_image is not None:
            # Save
            output_dir = Path("images/processed/simple")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{junocam_img.product_id}_simple_view.png"
            cv2.imwrite(str(output_file), output_image)
            print(f"\n✓ Saved: {output_file}")
        else:
            print("\n✗ Could not create view - Jupiter not visible in selected frame")

    finally:
        km.unload_kernels()
        print("\n✓ SPICE kernels unloaded")


if __name__ == "__main__":
    main()
