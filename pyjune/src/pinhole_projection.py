"""
Pinhole projection processing for JunoCam framelets.
Creates synthetic pinhole camera views by sampling framelets.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import spiceypy as spice
from dataclasses import dataclass

from .framelet_sampling import project_and_distort_jit, sample_framelet_at_positions
from .map_projection import JupiterEllipsoid


@dataclass
class Framelet:
    """Represents a single framelet (color band strip) from JunoCam."""

    frame_number: int
    color: str  # 'red', 'green', or 'blue'
    color_index: int  # 0=blue, 1=green, 2=red
    data: np.ndarray
    et: float = 0.0  # Ephemeris time for this framelet
    cam_position: np.ndarray = None  # Camera position in IAU_JUPITER frame
    cam_orient: np.ndarray = None  # Camera orientation matrix (JUNOCAM->IAU_JUPITER)

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]


def extract_framelets(
    fname: Path, start_et: float = 0.0, interframe_delay: float = 0.0
) -> dict:
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

        # Compute camera state and orientation for this frame
        # Use IAU_JUPITER frame so surface features stay at fixed coordinates over time
        state, _ = spice.spkezr("JUNO", frame_et, "IAU_JUPITER", "NONE", "JUPITER")
        cam_pos = state[:3]
        cam_orient = spice.pxform("JUNO_JUNOCAM", "IAU_JUPITER", frame_et)

        for color_idx in range(bands):
            base_row = frame_idx * frame_height + color_idx * band_height
            framelet_data = raw[base_row : base_row + band_height, :]

            framelet = Framelet(
                frame_number=frame_idx,
                color=color_map[color_idx],
                color_index=color_idx,
                data=framelet_data.copy(),
                et=frame_et,
                cam_position=cam_pos,
                cam_orient=cam_orient,
            )
            framelets_by_color[framelet.color].append(framelet)

    return framelets_by_color


def estimate_framelet_roi_fast(
    surface_positions: np.ndarray,
    framelet,
    camera_params,
    downsample_factor: int = 64,
) -> tuple:
    """
    Fast ROI estimation using simplified projection (no visibility/illumination checks).

    This lightweight version skips expensive surface normal computation and
    visibility checks, since we only need to find WHERE the framelet MIGHT
    project, not whether those points are actually valid.

    Args:
        surface_positions: Full grid of surface positions (H x W x 3)
        framelet: Framelet to estimate ROI for
        camera_params: Camera parameters
        downsample_factor: Factor to downsample by for coarse grid (default: 64)

    Returns:
        (y_min, y_max, x_min, x_max): ROI bounds in output grid coordinates,
        or None if no hits expected
    """

    h, w = surface_positions.shape[:2]

    # Create coarse grid by downsampling
    coarse_positions = surface_positions[::downsample_factor, ::downsample_factor, :]
    coarse_flat = coarse_positions.reshape(-1, 3)

    # Filter out NaN positions (not on planet surface)
    valid_surface = ~np.isnan(coarse_flat[:, 0])
    if not np.any(valid_surface):
        return None

    valid_pos = coarse_flat[valid_surface]

    # Transform to camera frame
    rays = valid_pos - framelet.cam_position
    rays_inst = rays @ framelet.cam_orient

    # Get camera parameters
    params = camera_params.get_params(framelet.color)
    focal_length = params["focal_length"]
    cx = params["cx"]
    cy = params["cy"]
    k1 = params["k1"]
    k2 = params["k2"]

    # Project to pixel coordinates (JIT-compiled)
    pixel_x, pixel_y = project_and_distort_jit(
        rays_inst, focal_length, cx, cy, k1, k2
    )

    # Check which pixels are in framelet bounds
    framelet_h, framelet_w = framelet.data.shape
    in_front = rays_inst[:, 2] > 0
    in_x_bounds = (pixel_x >= 0) & (pixel_x < framelet_w - 1)
    in_y_bounds = (pixel_y >= 0) & (pixel_y < framelet_h - 1)
    framelet_valid = in_front & in_x_bounds & in_y_bounds

    if not np.any(framelet_valid):
        return None  # No hits at all

    # Map back to coarse grid coordinates
    coarse_h, coarse_w = coarse_positions.shape[:2]
    hit_mask_flat = np.zeros(len(coarse_flat), dtype=bool)
    hit_mask_flat[valid_surface] = framelet_valid
    hit_mask = hit_mask_flat.reshape(coarse_h, coarse_w)

    # Find bounding box of hits in coarse grid
    hit_rows, hit_cols = np.where(hit_mask)

    # Convert back to full resolution coordinates with generous margin
    # Margin: 2× the downsample factor on each side
    margin = downsample_factor * 2

    y_min = max(0, hit_rows.min() * downsample_factor - margin)
    y_max = min(h, (hit_rows.max() + 1) * downsample_factor + margin)
    x_min = max(0, hit_cols.min() * downsample_factor - margin)
    x_max = min(w, (hit_cols.max() + 1) * downsample_factor + margin)

    return (y_min, y_max, x_min, x_max)


def project_framelets_to_pinhole_view(
    framelets_by_color: dict,
    ellipsoid: JupiterEllipsoid,
    camera_params,
    reference_framelet
):
    """
    Create a synthetic pinhole camera view by projecting framelets onto Jupiter's surface.
    Automatically sizes the view to capture all of Jupiter.

    Args:
        framelets_by_color: Dictionary of framelets by color
        ellipsoid: Jupiter ellipsoid model
        camera_params: CameraParameters instance with intrinsic camera parameters
        reference_framelet: Framelet to use for view geometry (typically middle frame)

    Returns:
        Tuple of (rgb_composite, red_channel, green_channel, blue_channel) or (None, None, None, None)
    """
    print("\n" + "=" * 70)
    print("CREATING SYNTHETIC PINHOLE VIEW")
    print("=" * 70)

    # Use precomputed camera state from reference framelet
    print(f"\n1. Getting camera state at ET={reference_framelet.et:.2f}...")
    cam_position = reference_framelet.cam_position
    cam_orient = reference_framelet.cam_orient

    print(f"   Camera position: {cam_position}")
    print(f"   Distance to Jupiter: {np.linalg.norm(cam_position):,.0f} km")

    # Get camera intrinsics from SPICE kernels
    focal_length_mm = spice.gdpool("INS-61500_FOCAL_LENGTH", 0, 1)[0]
    pixel_pitch_mm = spice.gdpool("INS-61500_PIXEL_SIZE", 0, 1)[0]
    focal_length_pixels = focal_length_mm / pixel_pitch_mm

    print(f"   Focal length: {focal_length_pixels:.1f} pixels")

    # Use the raw framelet dimensions directly
    print("   Calculating view size from framelet dimensions...")

    green_framelets = framelets_by_color["green"]
    num_framelets = len(
        [
            f
            for f in green_framelets
            if f.frame_number > 0 and f.frame_number < len(green_framelets) - 1
        ]
    )

    framelet_width = green_framelets[0].data.shape[1]  # 1648
    framelet_height = green_framelets[0].data.shape[0]  # 128

    # Total vertical extent of all framelets
    total_height = num_framelets * framelet_height

    # Use larger dimension for square output, with MORE margin
    view_size = int(
        max(framelet_width, total_height) * 2.0
    )  # Increased from 1.5 to 2.0

    # Clamp to reasonable size
    view_size = max(512, min(view_size, 4096))

    print(f"   Framelet dimensions: {framelet_width} x {framelet_height}")
    print(f"   Using {num_framelets} framelets")
    print(f"   Total vertical extent: {total_height} pixels")
    print(f"   View size: {view_size}x{view_size} pixels")

    # Get principal point for green band (detector center)
    cx = spice.gdpool("INS-61502_DISTORTION_X", 0, 1)[0]
    cy = spice.gdpool("INS-61502_DISTORTION_Y", 0, 1)[0]

    # Find where Jupiter actually appears by sampling a coarse grid
    print(f"\n2. Finding Jupiter's extent in detector space...")
    sample_size = 100  # Coarse grid for quick sampling

    # Create a large sampling grid to find Jupiter
    search_size = max(framelet_width, total_height) * 2
    search_half = search_size / 2

    y_sample, x_sample = np.mgrid[
        cy - search_half : cy + search_half : sample_size * 1j,
        cx - search_half : cx + search_half : sample_size * 1j,
    ]

    # Create rays for sampling
    rays_sample = np.stack(
        [
            x_sample.astype(np.float32),
            y_sample.astype(np.float32),
            np.full((sample_size, sample_size), focal_length_pixels, dtype=np.float32),
        ],
        axis=-1,
    )

    # Transform to Jupiter frame
    rays_jupiter_sample = rays_sample @ cam_orient.T

    # Test which rays hit Jupiter (vectorized)
    print(f"   Coarse sampling: testing {sample_size}×{sample_size} = {sample_size*sample_size} rays...")
    sample_intersections = ellipsoid.ray_intersection_vectorized(cam_position, rays_jupiter_sample)
    jupiter_hits = ~np.isnan(sample_intersections[..., 0])
    print(f"   Found {np.sum(jupiter_hits)} hits in coarse grid")

    # Find bounding box of Jupiter in detector space
    if np.any(jupiter_hits):
        hit_rows, hit_cols = np.where(jupiter_hits)
        y_min = y_sample[hit_rows.min(), 0]
        y_max = y_sample[hit_rows.max(), 0]
        x_min = x_sample[0, hit_cols.min()]
        x_max = x_sample[0, hit_cols.max()]

        # Add padding
        y_range = y_max - y_min
        x_range = x_max - x_min
        padding = 0.05  # %

        y_min -= y_range * padding
        y_max += y_range * padding
        x_min -= x_range * padding
        x_max += x_range * padding

        # Make it square by expanding the smaller dimension
        y_center = (y_min + y_max) / 2
        x_center = (x_min + x_max) / 2
        half_size = max(y_max - y_min, x_max - x_min) / 2

        y_start = y_center - half_size
        y_end = y_center + half_size
        x_start = x_center - half_size
        x_end = x_center + half_size

        print(
            f"   Jupiter extent: X=[{x_min:.0f}, {x_max:.0f}], Y=[{y_min:.0f}, {y_max:.0f}]"
        )
        print(f"   View center: ({x_center:.0f}, {y_center:.0f})")
        print(
            f"   View bounds: X=[{x_start:.0f}, {x_end:.0f}], Y=[{y_start:.0f}, {y_end:.0f}]"
        )
    else:
        # Fallback to centered view if Jupiter not found in sampling
        print("   Warning: Jupiter not found in coarse sampling, using centered view")
        half_size = view_size / 2
        x_start = cx - half_size
        x_end = cx + half_size
        y_start = cy - half_size
        y_end = cy + half_size

    print(f"\n3. Creating {view_size}x{view_size} view grid...")
    y, x = np.mgrid[y_start : y_end : view_size * 1j, x_start : x_end : view_size * 1j]

    # Create rays in camera pixel coordinates
    rays_camera = np.stack(
        [
            x.astype(np.float32),
            y.astype(np.float32),
            np.full((view_size, view_size), focal_length_pixels, dtype=np.float32),
        ],
        axis=-1,
    )

    # Transform rays to IAU_JUPITER frame
    # cam_orient transforms column vectors: v_jupiter = cam_orient @ v_inst
    # For row vectors, we need to transpose: v_jupiter = v_inst @ cam_orient.T
    rays_jupiter = rays_camera @ cam_orient.T

    # Project rays onto Jupiter surface
    print("\n4. Projecting rays onto Jupiter surface...")

    # Debug: test center pixel
    center_idx = view_size // 2
    center_ray = rays_jupiter[center_idx, center_idx]
    print(
        f"   Testing center pixel ({center_idx}, {center_idx}) -> detector pixel ({x[center_idx, center_idx]:.0f}, {y[center_idx, center_idx]:.0f})"
    )
    center_hit = ellipsoid.ray_intersection(cam_position, center_ray)
    print(f"   Center hit Jupiter: {center_hit is not None}")

    # VECTORIZED ray tracing - process all rays at once!
    print(f"   Vectorized ray tracing for {view_size:,}×{view_size:,} = {view_size*view_size:,} rays...")
    surface_positions = ellipsoid.ray_intersection_vectorized(cam_position, rays_jupiter)

    hit_count = np.sum(~np.isnan(surface_positions[..., 0]))

    valid_surface = ~np.isnan(surface_positions[..., 0])
    print(
        f"   Valid surface points: {np.sum(valid_surface):,} / {view_size*view_size:,} ({100*np.sum(valid_surface)/(view_size*view_size):.1f}%)"
    )

    # Sample framelets at surface positions
    print("\n5. Sampling framelets...")

    # Get Sun position (single query for entire image)
    sun_position, _ = spice.spkpos('SUN', reference_framelet.et, 'IAU_JUPITER', 'LT+S', 'JUPITER')
    print(f"   Sun position in IAU_JUPITER frame: {sun_position}")

    # Create arrays for RGB channels
    rgb_values = np.zeros((view_size, view_size, 3), dtype=np.float32)
    rgb_counts = np.zeros((view_size, view_size, 3), dtype=np.float32)

    # Process all three color channels
    color_names = ["red", "green", "blue"]
    color_indices = {"red": 0, "green": 1, "blue": 2}

    for color_name in color_names:
        framelets = framelets_by_color[color_name]
        channel_idx = color_indices[color_name]
        print(
            f"\n   Processing {color_name.upper()} channel ({len(framelets)} framelets)..."
        )

        for idx, framelet in enumerate(framelets):
            if (
                framelet.frame_number == 0
                or framelet.frame_number >= len(framelets) - 1
            ):
                continue

            # SPATIAL CULLING: Estimate which region could hit this framelet (fast version)
            roi = estimate_framelet_roi_fast(
                surface_positions,
                framelet,
                camera_params,
                downsample_factor=64,
            )

            if roi is None:
                # No hits expected - skip this framelet entirely
                if idx % 5 == 0:
                    print(f"      Frame {framelet.frame_number:3d}: 0 valid samples (skipped via ROI culling)")
                continue

            # Extract ROI from surface positions
            y_min, y_max, x_min, x_max = roi
            roi_positions = surface_positions[y_min:y_max, x_min:x_max, :]
            roi_size = roi_positions.shape[0] * roi_positions.shape[1]
            full_size = surface_positions.shape[0] * surface_positions.shape[1]

            # Sample framelet at ROI only
            sampled_roi, valid_mask_roi, debug_info = sample_framelet_at_positions(
                roi_positions,
                framelet.data,
                framelet.cam_position,
                framelet.cam_orient,
                ellipsoid,
                camera_params,
                sun_position,
                color=color_name,
            )

            # Place ROI results back into full arrays
            rgb_values[y_min:y_max, x_min:x_max, channel_idx] += sampled_roi
            rgb_counts[y_min:y_max, x_min:x_max, channel_idx] += valid_mask_roi

            # Update variables for logging
            sampled = sampled_roi
            valid_mask = valid_mask_roi

            if idx % 5 == 0:
                culling_pct = (1.0 - roi_size / full_size) * 100
                print(
                    f"      Frame {framelet.frame_number:3d}: {np.sum(valid_mask):,} valid samples "
                    f"(ROI: {roi_size:,}/{full_size:,} = {culling_pct:.1f}% culled)"
                )
                print(
                    f"        Validation: {debug_info['valid']}/{debug_info['total']} valid "
                    f"({debug_info['in_front']}/{debug_info['total']} in front, "
                    f"{debug_info['in_x']}/{debug_info['total']} in X, "
                    f"{debug_info['in_y']}/{debug_info['total']} in Y)"
                )
                print(
                    f"        Pixel ranges: X=[{debug_info['pixel_x_range'][0]:.1f}, {debug_info['pixel_x_range'][1]:.1f}] (valid: 0-{debug_info['framelet_size'][1]-1}), "
                    f"Y=[{debug_info['pixel_y_range'][0]:.1f}, {debug_info['pixel_y_range'][1]:.1f}] (valid: 0-{debug_info['framelet_size'][0]-1})"
                )

    # Check if we got any valid samples in any channel
    if rgb_counts.max() == 0:
        print("\n✗ No valid samples from any framelet!")
        return None, None, None, None

    # Average each channel independently
    print("\n6. Generating final RGB image...")
    for channel_idx in range(3):
        mask = rgb_counts[:, :, channel_idx] > 0
        rgb_values[mask, channel_idx] /= rgb_counts[mask, channel_idx]

    # Normalize each channel independently to 0-255
    rgb_normalized = np.zeros((view_size, view_size, 3), dtype=np.uint8)
    for channel_idx in range(3):
        channel_data = rgb_values[:, :, channel_idx]
        if channel_data.max() > 0:
            normalized = channel_data / channel_data.max() * 255
            rgb_normalized[:, :, channel_idx] = normalized.astype(np.uint8)

    # Set background (no data) to purple
    no_data_mask = rgb_counts.sum(axis=2) == 0
    rgb_normalized[no_data_mask, 0] = 128  # Red
    rgb_normalized[no_data_mask, 1] = 0    # Green
    rgb_normalized[no_data_mask, 2] = 128  # Blue

    # Create RGB composite (OpenCV uses BGR format)
    output_rgb = cv2.cvtColor(rgb_normalized, cv2.COLOR_RGB2BGR)

    # Extract individual channels
    output_red = rgb_normalized[:, :, 0]
    output_green = rgb_normalized[:, :, 1]
    output_blue = rgb_normalized[:, :, 2]

    return output_rgb, output_red, output_green, output_blue
