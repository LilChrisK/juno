"""
Framelet sampling utilities for JunoCam image processing.

Provides the core backward sampling function that projects surface positions
back to framelet pixel coordinates and interpolates pixel values.

Performance Note:
- Uses custom bilinear interpolation instead of scipy.interpolate.RectBivariateSpline
- This provides 10-30× speedup (from ~100ms to ~1ms per framelet)
"""

import numpy as np
import spiceypy as spice
import time
from typing import Tuple, Dict, Any

# Try to import numba for JIT compilation
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback: no-op decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class CameraParameters:
    """
    JunoCam camera parameters loaded from SPICE kernels.

    Stores intrinsic camera parameters (focal length, distortion coefficients)
    for all three color bands. These are instrument constants that never change,
    so we load them once to avoid redundant SPICE queries.

    Color band NAIF IDs:
    - Red:   -61503
    - Green: -61502
    - Blue:  -61501
    """

    def __init__(self):
        """Initialize camera parameters by querying SPICE kernels."""
        self.color_to_naif = {"red": -61503, "green": -61502, "blue": -61501}
        self.params = {}

        # Query parameters for all color bands
        for color, naif_id in self.color_to_naif.items():
            focal_length_mm = spice.gdpool(f"INS{naif_id}_FOCAL_LENGTH", 0, 1)[0]
            pixel_pitch_mm = spice.gdpool(f"INS{naif_id}_PIXEL_SIZE", 0, 1)[0]
            focal_length = focal_length_mm / pixel_pitch_mm

            cx = spice.gdpool(f"INS{naif_id}_DISTORTION_X", 0, 1)[0]
            cy = spice.gdpool(f"INS{naif_id}_DISTORTION_Y", 0, 1)[0]

            k1 = spice.gdpool(f"INS{naif_id}_DISTORTION_K1", 0, 1)[0]
            k2 = spice.gdpool(f"INS{naif_id}_DISTORTION_K2", 0, 1)[0]

            self.params[naif_id] = {
                "focal_length": focal_length,
                "cx": cx,
                "cy": cy,
                "k1": k1,
                "k2": k2,
            }

        print(f"Loaded camera parameters for {len(self.params)} color bands")
        if HAS_NUMBA:
            print("Using Numba JIT for projection/distortion (expect 2-3× speedup)")
        else:
            print("Numba not available - using numpy fallback")

    def get_params(self, color: str) -> dict:
        """
        Get camera parameters for a color band.

        Args:
            color: Color name ('red', 'green', or 'blue')

        Returns:
            Dictionary with focal_length, cx, cy, k1, k2
        """
        naif_id = self.color_to_naif.get(color.lower(), -61502)
        return self.params[naif_id]


@njit(fastmath=True, cache=True)
def project_and_distort_jit(rays_inst, focal_length, cx, cy, k1, k2):
    """
    JIT-compiled pinhole projection with radial distortion.

    Applies the JunoCam distortion model:
    1. Normalize by Z (perspective division)
    2. Apply radial distortion (dr = 1 + k1*r^2 + k2*r^4)
    3. Scale by focal length and add principal point

    Args:
        rays_inst: Ray directions in camera frame (N x 3)
        focal_length: Camera focal length in pixels
        cx, cy: Principal point coordinates
        k1, k2: Radial distortion coefficients

    Returns:
        pixel_x, pixel_y: Projected pixel coordinates (N,)
    """
    n = rays_inst.shape[0]
    pixel_x = np.empty(n, dtype=np.float64)
    pixel_y = np.empty(n, dtype=np.float64)

    for i in range(n):
        z = rays_inst[i, 2]
        if z != 0.0:
            inv_z = 1.0 / z
            cam_x = rays_inst[i, 0] * inv_z
            cam_y = rays_inst[i, 1] * inv_z

            # Apply radial distortion
            r2 = cam_x * cam_x + cam_y * cam_y
            dr_focal = (1.0 + k1 * r2 + k2 * r2 * r2) * focal_length

            pixel_x[i] = cam_x * dr_focal + cx
            pixel_y[i] = cam_y * dr_focal + cy
        else:
            # Invalid point (z=0)
            pixel_x[i] = np.nan
            pixel_y[i] = np.nan

    return pixel_x, pixel_y


@njit(fastmath=True, cache=True)
def process_visibility_and_transform_jit(
    valid_pos, cam_pos, cam_orient, sun_position, inv_a_sq, inv_c_sq
):
    """
    JIT-compiled hot path: surface normals + visibility + camera transform.

    Combines three expensive operations into one tight loop:
    1. Compute surface normals (ellipsoid geometry)
    2. Check visibility (camera-facing) and illumination (sun-lit)
    3. Transform rays to camera frame

    Args:
        valid_pos: Surface positions (N x 3)
        cam_pos: Camera position (3,)
        cam_orient: Camera orientation matrix (3 x 3)
        sun_position: Sun position (3,)
        inv_a_sq: 1 / (equatorial_radius^2)
        inv_c_sq: 1 / (polar_radius^2)

    Returns:
        rays_inst: Rays in camera frame (M x 3) for visible/illuminated points
        surface_valid: Boolean mask (N,) of which points passed checks
    """
    n = valid_pos.shape[0]
    surface_valid = np.zeros(n, dtype=np.bool_)
    rays_inst_list = []

    for i in range(n):
        # Compute surface normal for ellipsoid
        nx = valid_pos[i, 0] * inv_a_sq
        ny = valid_pos[i, 1] * inv_a_sq
        nz = valid_pos[i, 2] * inv_c_sq

        # Normalize
        norm = np.sqrt(nx * nx + ny * ny + nz * nz)
        nx /= norm
        ny /= norm
        nz /= norm

        # Compute rays
        ray_x = valid_pos[i, 0] - cam_pos[0]
        ray_y = valid_pos[i, 1] - cam_pos[1]
        ray_z = valid_pos[i, 2] - cam_pos[2]

        to_sun_x = sun_position[0] - valid_pos[i, 0]
        to_sun_y = sun_position[1] - valid_pos[i, 1]
        to_sun_z = sun_position[2] - valid_pos[i, 2]

        # Visibility check: normal points toward camera
        camera_dot = -(nx * ray_x + ny * ray_y + nz * ray_z)

        # Illumination check: normal points toward sun
        sun_dot = nx * to_sun_x + ny * to_sun_y + nz * to_sun_z

        # Both must be positive
        if camera_dot > 0 and sun_dot > 0:
            surface_valid[i] = True

            # Transform ray to camera frame: ray_inst = ray @ cam_orient
            rx = ray_x * cam_orient[0, 0] + ray_y * cam_orient[1, 0] + ray_z * cam_orient[2, 0]
            ry = ray_x * cam_orient[0, 1] + ray_y * cam_orient[1, 1] + ray_z * cam_orient[2, 1]
            rz = ray_x * cam_orient[0, 2] + ray_y * cam_orient[1, 2] + ray_z * cam_orient[2, 2]

            rays_inst_list.append((rx, ry, rz))

    # Convert list to array
    m = len(rays_inst_list)
    rays_inst = np.empty((m, 3), dtype=np.float64)
    for i in range(m):
        rays_inst[i, 0] = rays_inst_list[i][0]
        rays_inst[i, 1] = rays_inst_list[i][1]
        rays_inst[i, 2] = rays_inst_list[i][2]

    return rays_inst, surface_valid


@njit(fastmath=True, cache=True)
def bilinear_interp_jit(pixel_x, pixel_y, framelet_data, width, height):
    """
    JIT-compiled bilinear interpolation.

    Combines setup and computation into one tight loop for speed.

    Args:
        pixel_x: X coordinates (N,)
        pixel_y: Y coordinates (N,)
        framelet_data: Image data (height x width)
        width: Image width
        height: Image height

    Returns:
        sampled: Interpolated values (N,)
    """
    n = pixel_x.shape[0]
    sampled = np.empty(n, dtype=np.float64)

    for i in range(n):
        px = pixel_x[i]
        py = pixel_y[i]

        # Extract integer and fractional parts
        x0 = int(np.floor(px))
        y0 = int(np.floor(py))
        fx = px - x0
        fy = py - y0

        # Clip to bounds
        x0_clip = max(0, min(x0, width - 1))
        x1_clip = max(0, min(x0 + 1, width - 1))
        y0_clip = max(0, min(y0, height - 1))
        y1_clip = max(0, min(y0 + 1, height - 1))

        # Bilinear interpolation weights
        fx_inv = 1.0 - fx
        fy_inv = 1.0 - fy

        # Sample 4 corners
        f00 = framelet_data[y0_clip, x0_clip]
        f10 = framelet_data[y0_clip, x1_clip]
        f01 = framelet_data[y1_clip, x0_clip]
        f11 = framelet_data[y1_clip, x1_clip]

        # Weighted average
        sampled[i] = (
            f00 * fx_inv * fy_inv
            + f10 * fx * fy_inv
            + f01 * fx_inv * fy
            + f11 * fx * fy
        )

    return sampled


def sample_framelet_at_positions(
    surface_positions: np.ndarray,
    framelet_data: np.ndarray,
    cam_pos: np.ndarray,
    cam_orient: np.ndarray,
    ellipsoid,
    camera_params: CameraParameters,
    sun_position: np.ndarray,
    color: str = "green",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Sample framelet at given surface positions using backward projection.

    This is the core sampling function used by both pinhole and cylindrical
    projections. It takes surface positions and determines which framelet
    pixels they correspond to, then interpolates the pixel values.

    Args:
        surface_positions: Surface positions to sample (... x 3) in IAU_JUPITER frame
        framelet_data: Framelet pixel data (height x width)
        cam_pos: Camera position in IAU_JUPITER frame (km)
        cam_orient: Camera orientation matrix (JUNO_JUNOCAM -> IAU_JUPITER)
        ellipsoid: JupiterEllipsoid instance for computing surface normals
        camera_params: CameraParameters instance with intrinsic camera parameters
        sun_position: Sun position in IAU_JUPITER frame (km)
        color: Color band ('red', 'green', or 'blue')

    Returns:
        Tuple of (pixel_values, valid_mask, debug_info):
            - pixel_values: Interpolated brightness values (same shape as surface_positions[..., 0])
            - valid_mask: Boolean mask (1.0/0.0) of valid samples
            - debug_info: Dictionary with sampling statistics
    """
    # ========== TIMING SETUP ==========
    t_start = time.perf_counter()
    timings = {}

    # ========== SETUP ==========
    t0 = time.perf_counter()
    height, width = framelet_data.shape
    output_shape = surface_positions.shape[:-1]

    # Flatten
    flat_pos = surface_positions.reshape(-1, 3)
    valid_surface = ~np.isnan(flat_pos[:, 0])

    # DIAGNOSTIC: Show input array size and surface hit rate (commented out - too verbose with ROI culling)
    # input_size = len(flat_pos)
    # surface_hits = np.sum(valid_surface)
    # print(f"         [DIAGNOSTIC] Input: {input_size:,} positions ({output_shape}), Surface hits: {surface_hits:,} ({surface_hits/input_size*100:.1f}%)")

    pixel_values = np.zeros(len(flat_pos), dtype=np.float32)
    valid_mask = np.zeros(len(flat_pos), dtype=np.float32)
    timings["1_setup"] = (time.perf_counter() - t0) * 1000  # milliseconds

    if not np.any(valid_surface):
        debug_info = {
            "total": 0,
            "in_front": 0,
            "in_x": 0,
            "in_y": 0,
            "valid": 0,
            "pixel_x_range": (0, 0),
            "pixel_y_range": (0, 0),
            "framelet_size": (height, width),
        }
        return (
            pixel_values.reshape(output_shape),
            valid_mask.reshape(output_shape),
            debug_info,
        )

    # Get valid positions
    valid_pos = flat_pos[valid_surface]

    # ========== JIT-COMPILED HOT PATH: NORMALS + VISIBILITY + TRANSFORM ==========
    t0 = time.perf_counter()
    # Compute ellipsoid parameters once
    inv_a_sq = 1.0 / (ellipsoid.equatorial_radius_a ** 2)
    inv_c_sq = 1.0 / (ellipsoid.polar_radius ** 2)

    # Call JIT function that combines:
    # 1. Surface normals computation
    # 2. Visibility check (camera-facing)
    # 3. Illumination check (sun-lit)
    # 4. Camera frame transformation
    rays_inst, surface_valid_filtered = process_visibility_and_transform_jit(
        valid_pos, cam_pos, cam_orient, sun_position, inv_a_sq, inv_c_sq
    )

    # Update valid_surface mask to reflect filtered points
    idx = np.flatnonzero(valid_surface)
    valid_surface[:] = False
    valid_surface[idx[surface_valid_filtered]] = True

    timings["2_jit_hot_path"] = (time.perf_counter() - t0) * 1000

    if not np.any(valid_surface):
        debug_info = {
            "total": 0,
            "in_front": 0,
            "in_x": 0,
            "in_y": 0,
            "valid": 0,
            "pixel_x_range": (0, 0),
            "pixel_y_range": (0, 0),
            "framelet_size": (height, width),
        }
        return (
            pixel_values.reshape(output_shape),
            valid_mask.reshape(output_shape),
            debug_info,
        )

    # ========== GET CAMERA PARAMETERS ==========
    t0 = time.perf_counter()
    # Get camera parameters for this color band
    params = camera_params.get_params(color)
    focal_length = params["focal_length"]
    cx = params["cx"]
    cy = params["cy"]
    k1 = params["k1"]
    k2 = params["k2"]
    timings["6_get_camera_params"] = (time.perf_counter() - t0) * 1000

    # ========== PINHOLE PROJECTION + DISTORTION ==========
    t0 = time.perf_counter()
    # Apply distortion-corrected pinhole projection using JIT-compiled function
    # Per juno_junocam_v03.ti kernel documentation (lines 386-394):
    # 1. Normalize by Z (perspective division without focal length)
    # 2. Apply radial distortion model
    # 3. Scale by focal length and add principal point
    pixel_x, pixel_y = project_and_distort_jit(
        rays_inst, focal_length, cx, cy, k1, k2
    )
    timings["7_projection_distortion"] = (time.perf_counter() - t0) * 1000

    # ========== BOUNDS CHECKING ==========
    t0 = time.perf_counter()
    # Debug info
    in_front = rays_inst[:, 2] > 0
    in_x_bounds = (pixel_x >= 0) & (pixel_x < width - 1)
    in_y_bounds = (pixel_y >= 0) & (pixel_y < height - 1)

    # Check valid pixels
    framelet_valid = in_x_bounds & in_y_bounds & in_front
    timings["8_bounds_checking"] = (time.perf_counter() - t0) * 1000

    # Debug - return info about why validation failed
    num_valid = np.sum(framelet_valid)
    debug_info = {
        "total": len(valid_pos),
        "in_front": np.sum(in_front),
        "in_x": np.sum(in_x_bounds),
        "in_y": np.sum(in_y_bounds),
        "valid": num_valid,
        "pixel_x_range": (
            (float(pixel_x.min()), float(pixel_x.max())) if len(pixel_x) > 0 else (0, 0)
        ),
        "pixel_y_range": (
            (float(pixel_y.min()), float(pixel_y.max())) if len(pixel_y) > 0 else (0, 0)
        ),
        "framelet_size": (height, width),
    }

    # DIAGNOSTIC: Show final hit rate (commented out - too verbose with ROI culling)
    # print(f"         [DIAGNOSTIC] After filtering: {len(valid_pos):,} candidates -> {num_valid:,} valid samples ({num_valid/input_size*100:.3f}% of input)")


    if np.any(framelet_valid):
        # ========== JIT-COMPILED BILINEAR INTERPOLATION ==========
        t0 = time.perf_counter()
        # Extract valid coordinates
        valid_px = pixel_x[framelet_valid]
        valid_py = pixel_y[framelet_valid]

        # Call JIT-compiled interpolation (combines setup + compute)
        sampled = bilinear_interp_jit(valid_px, valid_py, framelet_data, width, height)
        timings["9_jit_interpolation"] = (time.perf_counter() - t0) * 1000

        # ========== MAP BACK TO OUTPUT ARRAYS ==========
        t0 = time.perf_counter()
        # Direct mapping without temporary arrays (optimized)
        # valid_surface tracks which flat_pos indices survived all filtering
        # framelet_valid is a mask on the filtered set indicating which passed bounds check
        base_indices = np.where(valid_surface)[0]
        final_indices = base_indices[framelet_valid]

        pixel_values[final_indices] = sampled
        valid_mask[final_indices] = 1.0
        timings["11_array_mapping"] = (time.perf_counter() - t0) * 1000

    # Calculate total time
    total_time = (time.perf_counter() - t_start) * 1000

    # Print timing breakdown
    print(f"      [TIMING] sample_framelet_at_positions: {total_time:.3f}ms total")
    print("        ┌─ Breakdown:")
    for key in sorted(timings.keys()):
        pct = (timings[key] / total_time * 100) if total_time > 0 else 0
        print(f"        │  {key:25s}: {timings[key]:6.3f}ms ({pct:5.1f}%)")
    print(f"        └─ Total: {total_time:.3f}ms")

    return (
        pixel_values.reshape(output_shape),
        valid_mask.reshape(output_shape),
        debug_info,
    )
