"""
Analyze features in cylindrical projections.

Load a cylindrical projection and convert pixel coordinates to lat/lon.
Interactive tool for feature identification.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.cylindrical_projection import CylindricalProjection
from src.projection_utils import load_projection_with_images, query_pixel_coordinates


# ============================================================================
# CONFIGURATION 
# ============================================================================

# Product ID to analyze (e.g., 'JNCE_2021159_34C00048_V01' or 'combined')
PRODUCT_ID = "JNCE_2021159_34C00080_V01"

# Number of random test points to display
NUM_POINTS = 5

# Show lat/lon grid overlay
SHOW_GRID = True

# Data directory
DATA_DIR = Path("images/processed/cylindrical")

# ============================================================================


def show_with_points(projection: CylindricalProjection, rgb_image: np.ndarray,
                     points: list[tuple[int, int]], product_id: str, show_grid: bool = True):
    """
    Display image with marked points and their coordinates.

    Args:
        projection: CylindricalProjection instance
        rgb_image: RGB image array
        points: List of (x, y) pixel coordinates
        product_id: Product ID for display
        show_grid: If True, show lat/lon grid overlay
    """
    # Convert BGR to RGB
    rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    # Calculate figure size based on projection aspect ratio
    aspect_ratio = projection.width / projection.height
    base_height = 8  # Base height in inches
    fig_width = base_height * aspect_ratio
    # Cap maximum width to avoid extremely wide windows
    fig_width = min(fig_width, 20)

    fig, ax = plt.subplots(figsize=(fig_width, base_height))
    ax.imshow(rgb, extent=[projection.lon_min, projection.lon_max,
                           projection.lat_min, projection.lat_max],
              aspect='auto', origin='upper')

    # Mark points
    for i, (x, y) in enumerate(points):
        # Use the existing query function
        result = query_pixel_coordinates(projection, x, y)

        if result['valid']:
            lat = result['latitude_deg']
            lon = result['longitude_deg']

            # Convert to extent coordinates for plotting
            ax.plot(lon, lat, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)

            # Add label
            label = f"{i+1}: ({lat:.1f}°, {lon:.1f}°W)"
            ax.annotate(label, xy=(lon, lat), xytext=(10, 10),
                       textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8),
                       fontsize=10, fontweight='bold')

            # Print to console
            print(f"Point {i+1}: pixel=({x:4d}, {y:4d}) -> lat={lat:7.2f}°, lon={lon:7.2f}°W")

    # Add grid
    if show_grid:
        # Latitude lines (horizontal)
        lat_spacing = 30  # degrees
        for lat_line in range(int(projection.lat_min), int(projection.lat_max) + 1, lat_spacing):
            if projection.lat_min <= lat_line <= projection.lat_max:
                ax.axhline(lat_line, color='white', alpha=0.3, linewidth=0.5, linestyle='--')
                ax.text(projection.lon_min + 5, lat_line, f"{lat_line}°",
                       color='white', fontsize=8, va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.5))

        # Longitude lines (vertical)
        lon_spacing = 30  # degrees
        for lon_line in range(int(projection.lon_min), int(projection.lon_max) + 1, lon_spacing):
            if projection.lon_min <= lon_line <= projection.lon_max:
                ax.axvline(lon_line, color='white', alpha=0.3, linewidth=0.5, linestyle='--')
                ax.text(lon_line, projection.lat_max - 5, f"{lon_line}°W",
                       color='white', fontsize=8, ha='left', rotation=90,
                       bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.5))

    ax.set_xlabel('Longitude (degrees West)', fontsize=12)
    ax.set_ylabel('Latitude (degrees)', fontsize=12)
    ax.set_title(f'Cylindrical Projection: {product_id}\n'
                f'Feature Analysis', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()


def main():
    """Main entry point."""
    print("=" * 70)
    print("CYLINDRICAL PROJECTION ANALYZER")
    print("=" * 70)
    print()

    # Load projection using existing utility
    metadata_path = DATA_DIR / f"{PRODUCT_ID}_cylindrical_metadata.json"

    if not metadata_path.exists():
        print(f"Error: Metadata file not found: {metadata_path}")
        return

    projection, rgb_image, _, _, _ = load_projection_with_images(metadata_path)

    print(f"Loaded: {PRODUCT_ID}")
    print(f"  Dimensions: {projection.width} x {projection.height} pixels")
    print(f"  Longitude range: {projection.lon_min}° to {projection.lon_max}° West")
    print(f"  Latitude range: {projection.lat_min}° to {projection.lat_max}°")
    print(f"  Resolution: {projection.resolution_deg}° per pixel")

    print()
    print("=" * 70)
    print("RANDOM TEST POINTS")
    print("=" * 70)

    # Generate random points
    np.random.seed(42)  # For reproducibility

    random_points = []
    for _ in range(NUM_POINTS):
        x = np.random.randint(0, projection.width)
        y = np.random.randint(0, projection.height)
        random_points.append((x, y))

    # Show image with points
    show_with_points(projection, rgb_image, random_points, PRODUCT_ID,
                     show_grid=SHOW_GRID)


if __name__ == "__main__":
    main()
