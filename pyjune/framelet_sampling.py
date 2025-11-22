"""
Framelet sampling utilities for JunoCam image processing.

Provides the core backward sampling function that projects surface positions
back to framelet pixel coordinates and interpolates pixel values.
"""

import numpy as np
import spiceypy as spice
from scipy.interpolate import RectBivariateSpline
from typing import Tuple, Dict, Any


def sample_framelet_at_positions(
    surface_positions: np.ndarray,
    framelet_data: np.ndarray,
    cam_pos: np.ndarray,
    cam_orient: np.ndarray,
    ellipsoid,
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
        color: Color band ('red', 'green', or 'blue')

    Returns:
        Tuple of (pixel_values, valid_mask, debug_info):
            - pixel_values: Interpolated brightness values (same shape as surface_positions[..., 0])
            - valid_mask: Boolean mask (1.0/0.0) of valid samples
            - debug_info: Dictionary with sampling statistics
    """
    # Map color to NAIF ID
    color_to_naif = {"red": -61503, "green": -61502, "blue": -61501}
    naif_id = color_to_naif.get(color.lower(), -61502)

    height, width = framelet_data.shape
    output_shape = surface_positions.shape[:-1]

    # Flatten
    flat_pos = surface_positions.reshape(-1, 3)
    valid_surface = ~np.isnan(flat_pos[:, 0])

    pixel_values = np.zeros(len(flat_pos), dtype=np.float32)
    valid_mask = np.zeros(len(flat_pos), dtype=np.float32)

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

    # Check visibility: surface normal must point towards camera
    # Compute surface normals (for oblate ellipsoid)
    a_sq = ellipsoid.equatorial_radius_a ** 2
    c_sq = ellipsoid.polar_radius ** 2

    # Surface normal for ellipsoid: gradient of (x/a)² + (y/a)² + (z/c)² = 1
    normals = np.zeros_like(valid_pos)
    normals[:, 0] = 2 * valid_pos[:, 0] / a_sq
    normals[:, 1] = 2 * valid_pos[:, 1] / a_sq
    normals[:, 2] = 2 * valid_pos[:, 2] / c_sq
    # Normalize
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # Vector from surface to camera
    to_camera = cam_pos - valid_pos
    to_camera_normalized = to_camera / np.linalg.norm(to_camera, axis=1, keepdims=True)

    # Dot product: positive means normal points towards camera (visible)
    dot_product = np.sum(normals * to_camera_normalized, axis=1)
    surface_visible = dot_product > 0

    # Filter to only visible surface points
    valid_pos = valid_pos[surface_visible]

    # Update valid_surface mask
    temp_mask = np.zeros(len(flat_pos), dtype=bool)
    temp_mask[valid_surface] = surface_visible
    valid_surface = temp_mask

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

    # Transform to camera frame
    rays = valid_pos - cam_pos
    # cam_orient transforms instrument->IAU_JUPITER: v_jupiter = cam_orient @ v_inst
    # So inverse is: v_inst = cam_orient.T @ v_jupiter (for column vectors)
    # For row vectors: v_inst = v_jupiter @ cam_orient (no transpose needed!)
    rays_inst = rays @ cam_orient

    # Use pinhole camera model with intrinsics from SPICE
    # Query band-specific parameters using NAIF ID
    focal_length_mm = spice.gdpool(f"INS{naif_id}_FOCAL_LENGTH", 0, 1)[0]
    pixel_pitch_mm = spice.gdpool(f"INS{naif_id}_PIXEL_SIZE", 0, 1)[0]
    focal_length = focal_length_mm / pixel_pitch_mm

    # Principal point for this color band
    cx = spice.gdpool(f"INS{naif_id}_DISTORTION_X", 0, 1)[0]
    cy = spice.gdpool(f"INS{naif_id}_DISTORTION_Y", 0, 1)[0]

    # Distortion coefficients (radial distortion model)
    k1 = spice.gdpool(f"INS{naif_id}_DISTORTION_K1", 0, 1)[0]
    k2 = spice.gdpool(f"INS{naif_id}_DISTORTION_K2", 0, 1)[0]

    # Apply distortion-corrected pinhole projection
    # Per juno_junocam_v03.ti kernel documentation (lines 386-394):
    # 1. Normalize by Z (perspective division without focal length)
    # 2. Apply radial distortion model
    # 3. Scale by focal length and add principal point
    with np.errstate(divide="ignore", invalid="ignore"):
        cam_x = rays_inst[:, 0] / rays_inst[:, 2]
        cam_y = rays_inst[:, 1] / rays_inst[:, 2]

        # Apply radial distortion: dr = 1 + k1*r^2 + k2*r^4
        r2 = cam_x**2 + cam_y**2
        dr = 1.0 + k1 * r2 + k2 * r2 * r2
        cam_x_distorted = cam_x * dr
        cam_y_distorted = cam_y * dr

        # Scale by focal length and add principal point
        pixel_x = cam_x_distorted * focal_length + cx
        pixel_y = cam_y_distorted * focal_length + cy

    # Debug info
    in_front = rays_inst[:, 2] > 0
    in_x_bounds = (pixel_x >= 0) & (pixel_x < width - 1)
    in_y_bounds = (pixel_y >= 0) & (pixel_y < height - 1)

    # Check valid pixels
    framelet_valid = in_x_bounds & in_y_bounds & in_front

    # Debug - return info about why validation failed
    debug_info = {
        "total": len(valid_pos),
        "in_front": np.sum(in_front),
        "in_x": np.sum(in_x_bounds),
        "in_y": np.sum(in_y_bounds),
        "valid": np.sum(framelet_valid),
        "pixel_x_range": (
            (float(pixel_x.min()), float(pixel_x.max())) if len(pixel_x) > 0 else (0, 0)
        ),
        "pixel_y_range": (
            (float(pixel_y.min()), float(pixel_y.max())) if len(pixel_y) > 0 else (0, 0)
        ),
        "framelet_size": (height, width),
    }

    if np.any(framelet_valid):
        # Interpolate
        interp_func = RectBivariateSpline(
            np.arange(height), np.arange(width), framelet_data, kx=1, ky=1
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

    return (
        pixel_values.reshape(output_shape),
        valid_mask.reshape(output_shape),
        debug_info,
    )
