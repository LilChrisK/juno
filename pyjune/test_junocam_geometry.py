"""
Test script to understand JunoCam geometry from SPICE kernels.
"""

import spiceypy as spice
import numpy as np
from spice_correction import SpiceKernelManager

def explore_junocam_geometry():
    """Explore JunoCam camera geometry from IK kernel."""

    km = SpiceKernelManager()
    km.load_kernels()

    try:
        print("=" * 70)
        print("JUNOCAM GEOMETRY FROM SPICE")
        print("=" * 70)

        # JunoCam NAIF ID
        camid = -61500

        # Get FOV information
        print("\n1. Field of View (FOV) Information:")
        print("-" * 70)
        shape, frame_name, boresight, n_bounds, bounds = spice.getfov(camid, 100)

        print(f"Shape: {shape}")
        print(f"Frame name: {frame_name}")
        print(f"Number of boundary vectors: {n_bounds}")
        print(f"Boresight (instrument frame): {boresight}")

        print(f"\nFOV Boundary Vectors (first 10):")
        for i in range(min(10, n_bounds)):
            print(f"  [{i:2d}]: [{bounds[i,0]:8.5f}, {bounds[i,1]:8.5f}, {bounds[i,2]:8.5f}]")

        # Calculate FOV angles
        print(f"\nFOV Angular Extent:")
        # Cross-track (X direction)
        x_angles = [np.arctan2(bounds[i,0], bounds[i,2]) for i in range(n_bounds)]
        x_min = np.degrees(min(x_angles))
        x_max = np.degrees(max(x_angles))
        print(f"  Cross-track (X): {x_min:.2f}° to {x_max:.2f}° (total: {x_max-x_min:.2f}°)")

        # Along-track (Y direction)
        y_angles = [np.arctan2(bounds[i,1], bounds[i,2]) for i in range(n_bounds)]
        y_min = np.degrees(min(y_angles))
        y_max = np.degrees(max(y_angles))
        print(f"  Along-track (Y): {y_min:.2f}° to {y_max:.2f}° (total: {y_max-y_min:.2f}°)")

        # Try to get specific camera parameters from kernel pool
        print("\n2. Camera Parameters from Kernel Pool:")
        print("-" * 70)

        try:
            # INS-61500_PIXEL_SAMPLES (width)
            pixel_samples = int(spice.gdpool(f'INS{camid}_PIXEL_SAMPLES', 0, 1)[0])
            print(f"Pixel samples (width): {pixel_samples}")
        except:
            print("Pixel samples: Not available")

        try:
            # INS-61500_PIXEL_LINES (height per framelet)
            pixel_lines = int(spice.gdpool(f'INS{camid}_PIXEL_LINES', 0, 1)[0])
            print(f"Pixel lines (height): {pixel_lines}")
        except:
            print("Pixel lines: Not available")

        try:
            # Pixel size
            pixel_size = spice.gdpool(f'INS{camid}_PIXEL_SIZE', 0, 1)[0]
            print(f"Pixel size: {pixel_size} mm")
        except:
            print("Pixel size: Not available")

        try:
            # Focal length
            focal_length = spice.gdpool(f'INS{camid}_FOCAL_LENGTH', 0, 1)[0]
            print(f"Focal length: {focal_length} mm")

            # Calculate pixel angular size
            if pixel_size:
                pixel_angle_rad = np.arctan(pixel_size / focal_length)
                pixel_angle_urad = pixel_angle_rad * 1e6
                print(f"Pixel angular size: {pixel_angle_urad:.1f} microradians")
        except:
            print("Focal length: Not available")

        try:
            # CCD center
            ccd_center = spice.gdpool(f'INS{camid}_CCD_CENTER', 0, 2)
            print(f"CCD center: {ccd_center}")
        except:
            print("CCD center: Not available")

        try:
            # Boresight sample/line
            boresight_sample = spice.gdpool(f'INS{camid}_BORESIGHT_SAMPLE', 0, 1)[0]
            boresight_line = spice.gdpool(f'INS{camid}_BORESIGHT_LINE', 0, 1)[0]
            print(f"Boresight pixel: (sample={boresight_sample}, line={boresight_line})")
        except:
            print("Boresight pixel: Not available")

        # Try to get pixel-to-angle transformation parameters
        print("\n3. Exploring Pixel-to-Angle Transformation:")
        print("-" * 70)

        # Test: What angle does pixel (0, 0) correspond to?
        # And pixel (1648, 0)? And (824, 64)?

        test_pixels = [
            (0, 0, "Top-left corner"),
            (1648, 0, "Top-right corner"),
            (824, 0, "Top-center"),
            (0, 128, "Bottom-left corner"),
            (1648, 128, "Bottom-right corner"),
            (824, 64, "Center"),
        ]

        # For a pushframe camera, we need to think differently
        # Each horizontal line is a 1D detector array
        # The "Y" dimension is built up over time as the spacecraft moves

        print("\nJunoCam is a PUSHFRAME camera:")
        print("  - 1D detector array with {pixel_samples} pixels wide")
        print("  - Sweeps across the scene as spacecraft moves")
        print("  - Each 'line' is captured at a different time")
        print("  - FOV is primarily cross-track (wide) with small along-track extent per line")

        # The key insight: for a single framelet line,
        # X position maps to cross-track angle
        # Y position (within a framelet) is mostly time, not angle

        print("\n4. Key Parameters for Projection:")
        print("-" * 70)
        cross_track_fov = x_max - x_min
        print(f"Cross-track FOV: {cross_track_fov:.2f}°")
        print(f"Per-pixel cross-track angle: {cross_track_fov / 1648:.4f}°/pixel")

        # For pushframe, the along-track FOV per instantaneous line is small
        instantaneous_fov_y = y_max - y_min
        print(f"Instantaneous along-track FOV: {instantaneous_fov_y:.4f}°")
        print(f"Per-pixel along-track angle: {instantaneous_fov_y / 128:.6f}°/pixel")

    finally:
        km.unload_kernels()


if __name__ == "__main__":
    explore_junocam_geometry()
