"""
Visualize Juno spacecraft geometry relative to Jupiter using SPICE.

Shows:
- Jupiter (as sphere)
- Juno position
- Velocity vector
- Camera pointing direction
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


def main():
    print("=" * 70)
    print("3D Visualization of Juno Geometry")
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

        # Get ephemeris time
        et = junocam_img.get_ephemeris_time()

        # Get spacecraft state (position and velocity)
        state, _ = spice.spkezr("JUNO", et, "J2000", "NONE", "JUPITER")

        # Position (km)
        pos = state[:3]
        # Velocity (km/s)
        vel = state[3:]

        # Get range and speed
        range_km = spice.vnorm(pos)
        speed_km_s = spice.vnorm(vel)

        print(f"\nSpacecraft State:")
        print(f"  Position (km): [{pos[0]:12.1f}, {pos[1]:12.1f}, {pos[2]:12.1f}]")
        print(f"  Velocity (km/s): [{vel[0]:8.3f}, {vel[1]:8.3f}, {vel[2]:8.3f}]")
        print(f"  Range: {range_km:,.1f} km")
        print(f"  Speed: {speed_km_s:.3f} km/s")

        # Get camera pointing (if CK available)
        try:
            # Get rotation from J2000 to JunoCam frame
            rotation = spice.pxform("J2000", "JUNO_JUNOCAM", et)

            # Camera boresight (pointing direction) in J2000
            # JunoCam boresight is typically along +Z in instrument frame
            camera_boresight_inst = np.array([0, 0, 1])

            # Transform to J2000
            camera_boresight_j2000 = rotation.T @ camera_boresight_inst

            # Scale for visualization (make it long to show pointing direction)
            camera_pointing = pos + camera_boresight_j2000 * 1000000  # 100,000 km arrow

            has_pointing = True
            print(f"  Camera boresight: [{camera_boresight_j2000[0]:8.3f}, {camera_boresight_j2000[1]:8.3f}, {camera_boresight_j2000[2]:8.3f}]")
        except:
            has_pointing = False
            print(f"  Camera pointing: (not available - need CK kernel)")

        # Create 3D plot
        print("\nCreating 3D visualization...")
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Jupiter radius (equatorial)
        jupiter_radius = 71492  # km

        # Plot Jupiter
        plot_sphere(ax, jupiter_radius, color='orange', alpha=0.3)

        # Plot spacecraft position
        ax.scatter(pos[0], pos[1], pos[2],
                  c='blue', s=100, marker='^',
                  label='Juno spacecraft')

        # Plot velocity vector
        vel_scale = 1000  # Scale velocity for visualization
        ax.quiver(pos[0], pos[1], pos[2],
                 vel[0], vel[1], vel[2],
                 length=vel_scale, color='green',
                 arrow_length_ratio=0.3, linewidth=2,
                 label='Velocity')

        # Plot camera pointing
        if has_pointing:
            ax.plot([pos[0], camera_pointing[0]],
                   [pos[1], camera_pointing[1]],
                   [pos[2], camera_pointing[2]],
                   'r-', linewidth=2, label='Camera boresight')

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
        ax.set_title(f'Juno Geometry at {junocam_img.image_time}\n'
                    f'Range: {range_km:,.0f} km, Speed: {speed_km_s:.2f} km/s',
                    fontsize=12, pad=20)
        ax.legend(loc='upper right')

        # Add grid
        ax.grid(True, alpha=0.3)

        # Adjust view angle
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()

        # Save figure
        output_file = "images/processed/geometry_3d.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved visualization: {output_file}")

        # Show plot
        plt.show()

    finally:
        km.unload_kernels()


if __name__ == "__main__":
    main()
