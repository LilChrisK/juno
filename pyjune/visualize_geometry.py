"""
Visualize Juno spacecraft geometry relative to Jupiter using SPICE.

Shows:
- Jupiter (as sphere)
- Juno position
- Velocity vector
- Camera boresight (center pointing)
- Camera FOV corners (4 vectors showing field of view extent)
- FOV boundary lines connecting the corners
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import spiceypy as spice
from pathlib import Path

from spice_correction import SpiceKernelManager, JunoCamImage


def plot_sphere(ax, radius, center=(0, 0, 0), color='orange', alpha=0.3, label='Jupiter'):
    """Plot a sphere (Jupiter)."""
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

    ax.plot_surface(x, y, z, color=color, alpha=alpha, label=label)


def plot_frame_geometry(ax, et, frame_num, junocam_img, jupiter_radius):
    """Plot geometry for a single frame."""
    # Get spacecraft state (position and velocity)
    state, _ = spice.spkezr("JUNO", et, "J2000", "NONE", "JUPITER")

    # Position (km)
    pos = state[:3]
    # Velocity (km/s)
    vel = state[3:]

    # Get range and speed
    range_km = spice.vnorm(pos)
    speed_km_s = spice.vnorm(vel)

    print(f"\nFrame {frame_num}:")
    print(f"  Position (km): [{pos[0]:12.1f}, {pos[1]:12.1f}, {pos[2]:12.1f}]")
    print(f"  Range: {range_km:,.1f} km")

    # Get camera pointing and FOV
    try:
        # Get rotation matrix from JUNO_JUNOCAM frame to J2000 frame
        rotation = spice.pxform("JUNO_JUNOCAM", "J2000", et)

        # Camera boresight (pointing direction) in instrument frame
        camera_boresight_inst = np.array([0, 0, 1])

        # Transform to J2000
        camera_boresight_j2000 = rotation @ camera_boresight_inst

        # Get FOV boundary corners
        result = spice.getfov(-61500, 16)
        fov_bounds = result[4]

        # Verify camera is pointing toward Jupiter
        jupiter_direction = -pos / np.linalg.norm(pos)
        pointing_angle = np.arccos(np.dot(camera_boresight_j2000, jupiter_direction))
        pointing_angle_deg = np.degrees(pointing_angle)

        print(f"  Camera boresight: [{camera_boresight_j2000[0]:8.3f}, {camera_boresight_j2000[1]:8.3f}, {camera_boresight_j2000[2]:8.3f}]")
        print(f"  Angle to Jupiter: {pointing_angle_deg:.2f}°")

        # Transform FOV corner vectors
        num_corners = fov_bounds.shape[0]
        corner_indices = [0, num_corners//4, num_corners//2, 3*num_corners//4]

        fov_corners_j2000 = []
        for i in corner_indices:
            corner_inst = fov_bounds[i, :]
            corner_j2000 = rotation @ corner_inst
            fov_corners_j2000.append(corner_j2000)

        fov_scale = 1000000  # 1,000,000 km
        has_pointing = True
    except Exception as e:
        has_pointing = False
        print(f"  Camera pointing: (error - {e})")

    # Plot Jupiter
    plot_sphere(ax, jupiter_radius, color='orange', alpha=0.3)

    # Plot spacecraft position
    ax.scatter(pos[0], pos[1], pos[2],
              c='blue', s=100, marker='^',
              label='Juno spacecraft')

    # Plot velocity vector
    vel_scale = 1000
    ax.quiver(pos[0], pos[1], pos[2],
             vel[0], vel[1], vel[2],
             length=vel_scale, color='green',
             arrow_length_ratio=0.3, linewidth=2,
             label='Velocity')

    # Plot camera FOV
    if has_pointing:
        # Plot boresight (center line)
        boresight_end = pos + camera_boresight_j2000 * fov_scale
        ax.plot([pos[0], boresight_end[0]],
               [pos[1], boresight_end[1]],
               [pos[2], boresight_end[2]],
               'r-', linewidth=2, label='Camera boresight', alpha=0.8)

        # Plot FOV corner vectors
        colors = ['red', 'yellow', 'cyan', 'magenta']
        for i, (corner, color) in enumerate(zip(fov_corners_j2000, colors)):
            corner_end = pos + corner * fov_scale
            ax.plot([pos[0], corner_end[0]],
                   [pos[1], corner_end[1]],
                   [pos[2], corner_end[2]],
                   color=color, linewidth=1.5, alpha=0.7,
                   label=f'FOV corner {i+1}' if i == 0 else '')

        # Connect the corners to show the FOV boundary
        for i in range(len(fov_corners_j2000)):
            j = (i + 1) % len(fov_corners_j2000)
            corner_i_end = pos + fov_corners_j2000[i] * fov_scale
            corner_j_end = pos + fov_corners_j2000[j] * fov_scale
            ax.plot([corner_i_end[0], corner_j_end[0]],
                   [corner_i_end[1], corner_j_end[1]],
                   [corner_i_end[2], corner_j_end[2]],
                   'r--', linewidth=1, alpha=0.4)

    # Plot trajectory from spacecraft to Jupiter center
    ax.plot([0, pos[0]], [0, pos[1]], [0, pos[2]],
           'k--', alpha=0.3, linewidth=1)

    # Set labels and limits
    ax.set_xlabel('X (km)', fontsize=10)
    ax.set_ylabel('Y (km)', fontsize=10)
    ax.set_zlabel('Z (km)', fontsize=10)

    # Set equal aspect ratio
    max_range = range_km * 1.2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    # Title and legend
    utc = spice.et2utc(et, "C", 3)
    ax.set_title(f'Frame {frame_num} at {utc}\n'
                f'Range: {range_km:,.0f} km, Speed: {speed_km_s:.2f} km/s',
                fontsize=12, pad=20)
    ax.legend(loc='upper right')

    # Add grid
    ax.grid(True, alpha=0.3)

    # Adjust view angle
    ax.view_init(elev=20, azim=45)


def main():
    print("=" * 70)
    print("3D Visualization of Juno Geometry - Per Frame")
    print("=" * 70)

    # Load SPICE kernels
    print("\nLoading SPICE kernels...")
    km = SpiceKernelManager()
    km.load_kernels()

    try:
        # Load image metadata
        fname = Path("images/raw/JNCE_2021159_34C00080_V01-raw.png")
        junocam_img = JunoCamImage(fname)

        print(f"\nImage: {junocam_img.product_id}")
        print(f"Date: {junocam_img.image_time}")

        # Get timing information from metadata
        if not junocam_img.metadata:
            print("ERROR: No metadata available")
            return

        # Parse interframe delay
        interframe_delay_str = junocam_img.metadata.get("INTERFRAME_DELAY", "0.371 <s>")
        interframe_delay = float(interframe_delay_str.split()[0])

        # Get start ephemeris time
        et_start = junocam_img.get_ephemeris_time()

        # Calculate number of frames
        band_height = 128
        bands = 3
        frame_height = band_height * bands

        # Read the raw image to get dimensions
        import cv2
        raw = cv2.imread(str(fname), cv2.IMREAD_UNCHANGED)
        if raw is None:
            print(f"Could not load image: {fname}")
            return

        num_frames = raw.shape[0] // frame_height

        print(f"\nTiming information:")
        print(f"  Start time: {junocam_img.metadata['START_TIME']}")
        print(f"  Stop time: {junocam_img.metadata['STOP_TIME']}")
        print(f"  Interframe delay: {interframe_delay} seconds")
        print(f"  Number of frames: {num_frames}")
        print(f"  Total imaging time: {interframe_delay * num_frames:.2f} seconds")

        # Jupiter radius (equatorial)
        jupiter_radius = 71492  # km

        print("\n" + "=" * 70)
        print("Close each plot window to see the next frame")
        print("Press Ctrl+C to stop")
        print("=" * 70)

        # Plot each frame
        for frame_num in range(num_frames):
            # Calculate ephemeris time for this frame
            et_frame = et_start + (frame_num * interframe_delay)

            # Create a new figure for each frame
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Plot this frame's geometry
            plot_frame_geometry(ax, et_frame, frame_num, junocam_img, jupiter_radius)

            plt.tight_layout()

            # Save frame
            output_file = f"images/processed/geometry_frame_{frame_num:02d}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"  Saved: {output_file}")

            # Show plot (blocks until window is closed)
            plt.show()

            # Close the figure to free memory
            plt.close(fig)

        print("\n✓ All frames visualized!")

    finally:
        km.unload_kernels()


if __name__ == "__main__":
    main()
