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

from src.spice_correction import SpiceKernelManager, JunoCamImage


def plot_sphere(
    ax, radius, center=(0, 0, 0), color="orange", alpha=0.3, label="Jupiter"
):
    """Plot a sphere (Jupiter)."""
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

    ax.plot_surface(x, y, z, color=color, alpha=alpha, label=label)


def ray_sphere_intersection(ray_origin, ray_direction, sphere_center, sphere_radius):
    """
    Calculate ray-sphere intersection point.

    Args:
        ray_origin: Starting point of ray (3D vector)
        ray_direction: Direction of ray (normalized 3D vector)
        sphere_center: Center of sphere (3D vector)
        sphere_radius: Radius of sphere (scalar)

    Returns:
        Intersection point (3D vector) or None if no intersection
    """
    # Vector from sphere center to ray origin
    oc = ray_origin - sphere_center

    # Quadratic equation coefficients: at² + bt + c = 0
    # Ray: P(t) = origin + t * direction
    # Sphere: |P - center|² = radius²
    a = np.dot(ray_direction, ray_direction)
    b = 2.0 * np.dot(oc, ray_direction)
    c = np.dot(oc, oc) - sphere_radius * sphere_radius

    discriminant = b * b - 4 * a * c

    # No intersection
    if discriminant < 0:
        return None

    # Calculate the two possible t values
    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    # We want the first positive intersection (closest to origin)
    t = None
    if t1 > 0:
        t = t1
    elif t2 > 0:
        t = t2

    if t is None:
        return None

    # Calculate intersection point
    intersection = ray_origin + t * ray_direction
    return intersection


def plot_fov_edge_intersection(
    ax,
    pos,
    corner1,
    corner2,
    sphere_center,
    sphere_radius,
    num_samples=50,
    color="lime",
    linewidth=2,
):
    """
    Plot the intersection curve between a FOV edge plane and Jupiter's surface.

    Samples points along the edge between two corners and projects them onto Jupiter.
    """
    edge_points = []

    # Sample points along the edge from corner1 to corner2
    for alpha in np.linspace(0, 1, num_samples):
        # Interpolate between corners (spherical interpolation would be more accurate, but linear is simpler)
        direction = (1 - alpha) * corner1 + alpha * corner2
        direction = direction / np.linalg.norm(direction)  # Normalize

        # Find intersection with Jupiter
        intersection = ray_sphere_intersection(
            pos, direction, sphere_center, sphere_radius
        )
        if intersection is not None:
            edge_points.append(intersection)

    # Draw the curve if we have points
    if len(edge_points) > 1:
        edge_points = np.array(edge_points)
        ax.plot(
            edge_points[:, 0],
            edge_points[:, 1],
            edge_points[:, 2],
            color=color,
            linewidth=linewidth,
            alpha=0.9,
        )
        return True
    return False


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
    print(f"  Velocity (km/s): [{vel[0]:8.3f}, {vel[1]:8.3f}, {vel[2]:8.3f}]")
    print(f"  Speed: {speed_km_s:.3f} km/s")

    # Get camera pointing and FOV
    try:
        # Get rotation matrix from JUNO_JUNOCAM frame to J2000 frame
        rotation = spice.pxform("JUNO_JUNOCAM", "J2000", et)

        # Get FOV information from SPICE kernel
        shape, frame_name, boresight_inst, n, fov_bounds = spice.getfov(-61500, 16)

        # Transform boresight to J2000
        camera_boresight_j2000 = rotation @ boresight_inst

        # Verify camera is pointing toward Jupiter
        jupiter_direction = -pos / np.linalg.norm(pos)
        pointing_angle = np.arccos(np.dot(camera_boresight_j2000, jupiter_direction))
        pointing_angle_deg = np.degrees(pointing_angle)

        print(
            f"  Camera boresight: [{camera_boresight_j2000[0]:8.3f}, {camera_boresight_j2000[1]:8.3f}, {camera_boresight_j2000[2]:8.3f}]"
        )
        print(f"  Angle to Jupiter: {pointing_angle_deg:.2f}°")

        # Transform FOV corner vectors
        num_corners = fov_bounds.shape[0]
        corner_indices = [0, num_corners // 4, num_corners // 2, 3 * num_corners // 4]

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
    plot_sphere(ax, jupiter_radius, color="orange", alpha=0.3)

    # Plot spacecraft position
    ax.scatter(
        pos[0], pos[1], pos[2], c="blue", s=100, marker="^", label="Juno spacecraft"
    )

    # Plot velocity vector
    vel_scale = 1000
    ax.quiver(
        pos[0],
        pos[1],
        pos[2],
        vel[0],
        vel[1],
        vel[2],
        length=vel_scale,
        color="green",
        arrow_length_ratio=0.3,
        linewidth=2,
        label="Velocity",
    )

    # Plot camera FOV
    if has_pointing:
        sphere_center = np.array([0.0, 0.0, 0.0])

        # Calculate boresight intersection with Jupiter
        boresight_intersection = ray_sphere_intersection(
            pos, camera_boresight_j2000, sphere_center, jupiter_radius
        )

        # Plot boresight (center line)
        if boresight_intersection is not None:
            # Ray hits Jupiter - draw to intersection point
            ax.plot(
                [pos[0], boresight_intersection[0]],
                [pos[1], boresight_intersection[1]],
                [pos[2], boresight_intersection[2]],
                "r-",
                linewidth=2,
                label="Camera boresight",
                alpha=0.8,
            )
            # Highlight intersection on sphere
            ax.scatter(
                boresight_intersection[0],
                boresight_intersection[1],
                boresight_intersection[2],
                c="red",
                s=200,
                marker="*",
                edgecolors="white",
                linewidths=2,
                label="Boresight on Jupiter",
                zorder=10,
            )
            print(
                f"  ✓ Boresight hits Jupiter at: [{boresight_intersection[0]:.1f}, {boresight_intersection[1]:.1f}, {boresight_intersection[2]:.1f}]"
            )
        else:
            # Ray misses - draw extended line
            boresight_end = pos + camera_boresight_j2000 * fov_scale
            ax.plot(
                [pos[0], boresight_end[0]],
                [pos[1], boresight_end[1]],
                [pos[2], boresight_end[2]],
                "r--",
                linewidth=2,
                label="Camera boresight (miss)",
                alpha=0.5,
            )
            print(f"  ✗ Boresight misses Jupiter")

        # Plot FOV corner vectors and intersections
        colors = ["red", "yellow", "cyan", "magenta"]
        intersection_points = []

        for i, (corner, color) in enumerate(zip(fov_corners_j2000, colors)):
            # Calculate intersection
            corner_intersection = ray_sphere_intersection(
                pos, corner, sphere_center, jupiter_radius
            )

            if corner_intersection is not None:
                # Ray hits Jupiter
                ax.plot(
                    [pos[0], corner_intersection[0]],
                    [pos[1], corner_intersection[1]],
                    [pos[2], corner_intersection[2]],
                    color=color,
                    linewidth=1.5,
                    alpha=0.7,
                )
                # Highlight on sphere
                ax.scatter(
                    corner_intersection[0],
                    corner_intersection[1],
                    corner_intersection[2],
                    c=color,
                    s=150,
                    marker="o",
                    edgecolors="white",
                    linewidths=1.5,
                    zorder=10,
                )
                intersection_points.append(corner_intersection)
            else:
                # Ray misses - draw extended line
                corner_end = pos + corner * fov_scale
                ax.plot(
                    [pos[0], corner_end[0]],
                    [pos[1], corner_end[1]],
                    [pos[2], corner_end[2]],
                    color=color,
                    linewidth=1.5,
                    alpha=0.3,
                    linestyle="--",
                )

        # Draw FOV edge intersection curves on Jupiter's surface
        if len(fov_corners_j2000) >= 4:
            edge_count = 0
            for i in range(len(fov_corners_j2000)):
                j = (i + 1) % len(fov_corners_j2000)
                # Draw the intersection curve for this edge
                has_edge = plot_fov_edge_intersection(
                    ax,
                    pos,
                    fov_corners_j2000[i],
                    fov_corners_j2000[j],
                    sphere_center,
                    jupiter_radius,
                    num_samples=50,
                    color="lime",
                    linewidth=3,
                )
                if has_edge:
                    edge_count += 1

            if edge_count > 0:
                print(f"  ✓ FOV footprint: {edge_count} edges on Jupiter surface")
                # Add a dummy line for the legend
                ax.plot([], [], "lime", linewidth=3, alpha=0.9, label="FOV footprint")

    # Plot trajectory from spacecraft to Jupiter center
    ax.plot([0, pos[0]], [0, pos[1]], [0, pos[2]], "k--", alpha=0.3, linewidth=1)

    # Set labels and limits
    ax.set_xlabel("X (km)", fontsize=10)
    ax.set_ylabel("Y (km)", fontsize=10)
    ax.set_zlabel("Z (km)", fontsize=10)

    # Set equal aspect ratio
    max_range = range_km * 1.2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    # Title and legend
    utc = spice.et2utc(et, "C", 3)
    ax.set_title(
        f"Frame {frame_num} at {utc}\n"
        f"Range: {range_km:,.0f} km, Speed: {speed_km_s:.2f} km/s",
        fontsize=12,
        pad=20,
    )
    ax.legend(loc="upper right")

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
        # fname = Path("images/raw/JNCE_2021159_34C00080_V01-raw.png")
        # fname = Path("images/raw/JNCE_2021159_34C00055_V01-raw.png")
        fname = Path("images/raw/JNCE_2021159_34C00048_V01-raw.png")
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

        # Create output directory
        output_dir = Path("images/processed/geo")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput directory: {output_dir}")

        print("\n" + "=" * 70)
        print("Generating geometry visualizations (no display)")
        print("Press Ctrl+C to stop")
        print("=" * 70)

        # Plot each frame
        for frame_num in range(num_frames):
            # Calculate ephemeris time for this frame
            et_frame = et_start + (frame_num * interframe_delay)

            # Create a new figure for each frame
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")

            # Plot this frame's geometry
            plot_frame_geometry(ax, et_frame, frame_num, junocam_img, jupiter_radius)

            plt.tight_layout()

            # Save frame
            output_file = output_dir / f"geometry_frame_{frame_num:02d}.png"
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            print(f"  Saved: {output_file}")

            if frame_num in [9, 14, 16]:
                plt.show()
            # Close the figure to free memory (no display)
            plt.close(fig)

        print("\n✓ All frames visualized!")

    finally:
        km.unload_kernels()


if __name__ == "__main__":
    main()
