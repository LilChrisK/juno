"""
Utility functions for working with cylindrical projections.

Provides convenience functions for loading, querying, and analyzing
saved projection maps.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from src.cylindrical_projection import CylindricalProjection


def load_projection_with_images(
    metadata_path: Path
) -> Tuple[CylindricalProjection, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a cylindrical projection with all associated images.

    Args:
        metadata_path: Path to metadata JSON file

    Returns:
        Tuple of (projection, rgb_image, red_channel, green_channel, blue_channel)
    """
    # Load projection metadata
    projection = CylindricalProjection.load(metadata_path)

    # Load images
    base_path = metadata_path.parent / metadata_path.stem.replace('_metadata', '')

    rgb = cv2.imread(str(base_path.parent / f"{base_path.stem}_rgb.png"))
    red = cv2.imread(str(base_path.parent / f"{base_path.stem}_red.png"), cv2.IMREAD_GRAYSCALE)
    green = cv2.imread(str(base_path.parent / f"{base_path.stem}_green.png"), cv2.IMREAD_GRAYSCALE)
    blue = cv2.imread(str(base_path.parent / f"{base_path.stem}_blue.png"), cv2.IMREAD_GRAYSCALE)

    if rgb is None or red is None or green is None or blue is None:
        raise FileNotFoundError(f"Could not load images for {metadata_path}")

    return projection, rgb, red, green, blue


def query_pixel_coordinates(
    projection: CylindricalProjection,
    pixel_x: int,
    pixel_y: int
) -> Dict[str, Any]:
    """
    Query the coordinates at a given pixel location.

    Args:
        projection: Cylindrical projection
        pixel_x: Pixel x-coordinate
        pixel_y: Pixel y-coordinate

    Returns:
        Dictionary with coordinate information
    """
    if not (0 <= pixel_x < projection.width and 0 <= pixel_y < projection.height):
        return {
            "valid": False,
            "error": "Pixel coordinates out of bounds"
        }

    lat, lon = projection.pixel_to_latlon(pixel_x, pixel_y)

    return {
        "valid": True,
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "latitude_deg": lat,
        "longitude_deg": lon,
        "longitude_system": "System III (1965) West"
    }


def query_latlon_pixel(
    projection: CylindricalProjection,
    latitude: float,
    longitude: float
) -> Dict[str, Any]:
    """
    Query the pixel location for given coordinates.

    Args:
        projection: Cylindrical projection
        latitude: Latitude in degrees
        longitude: Longitude in degrees (West)

    Returns:
        Dictionary with pixel information
    """
    px, py = projection.latlon_to_pixel(latitude, longitude)

    # Check if pixel is within bounds
    in_bounds = (0 <= px < projection.width and 0 <= py < projection.height)

    return {
        "valid": in_bounds,
        "latitude_deg": latitude,
        "longitude_deg": longitude,
        "pixel_x": px,
        "pixel_y": py,
        "pixel_x_int": int(round(px)) if in_bounds else None,
        "pixel_y_int": int(round(py)) if in_bounds else None
    }


def get_coverage_map(projection: CylindricalProjection) -> np.ndarray:
    """
    Get a boolean coverage map showing which pixels have data.

    Args:
        projection: Cylindrical projection

    Returns:
        Boolean array (height x width) where True = pixel has data
    """
    return projection.map_counts > 0


def get_projection_stats(projection: CylindricalProjection) -> Dict[str, Any]:
    """
    Get statistics about the projection.

    Args:
        projection: Cylindrical projection

    Returns:
        Dictionary with statistics
    """
    coverage_map = get_coverage_map(projection)
    total_pixels = projection.width * projection.height
    valid_pixels = np.sum(coverage_map)

    red, green, blue = projection.get_maps()

    return {
        "dimensions": {
            "width": projection.width,
            "height": projection.height,
            "total_pixels": total_pixels
        },
        "coverage": {
            "valid_pixels": int(valid_pixels),
            "coverage_percent": float(valid_pixels / total_pixels * 100)
        },
        "bounds": {
            "lon_min": projection.lon_min,
            "lon_max": projection.lon_max,
            "lat_min": projection.lat_min,
            "lat_max": projection.lat_max
        },
        "resolution_deg": projection.resolution_deg,
        "data_quality": {
            "max_samples_per_pixel": float(np.max(projection.map_counts)),
            "mean_samples_per_pixel": float(np.mean(projection.map_counts[coverage_map])) if valid_pixels > 0 else 0.0
        }
    }


def print_projection_info(metadata_path: Path):
    """
    Print detailed information about a saved projection.

    Args:
        metadata_path: Path to metadata JSON file
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print("=" * 70)
    print("CYLINDRICAL PROJECTION INFO")
    print("=" * 70)
    print(f"\nProduct: {metadata['product_id']}")
    print(f"Projection: {metadata['projection_type']}")
    print(f"Coordinate System: {metadata['coordinate_system']}")

    print(f"\nBounds:")
    print(f"  Longitude: {metadata['lon_min']:.1f}° to {metadata['lon_max']:.1f}° West")
    print(f"  Latitude: {metadata['lat_min']:.1f}° to {metadata['lat_max']:.1f}°")

    print(f"\nResolution:")
    print(f"  {metadata['resolution_deg']:.3f} degrees/pixel")
    print(f"  {metadata['width']} x {metadata['height']} pixels")

    print(f"\nCoverage:")
    print(f"  {metadata['coverage']['valid_pixels']:,} / {metadata['coverage']['total_pixels']:,} pixels")
    print(f"  {metadata['coverage']['coverage_percent']:.1f}%")

    if 'reference_et' in metadata and metadata['reference_et'] is not None:
        print(f"\nReference ET: {metadata['reference_et']:.2f}")

    print("=" * 70)


def create_annotated_map(
    projection: CylindricalProjection,
    rgb_image: np.ndarray,
    grid_spacing_deg: float = 30.0,
    output_path: Optional[Path] = None
) -> np.ndarray:
    """
    Create an annotated map with lat/lon grid lines.

    Args:
        projection: Cylindrical projection
        rgb_image: RGB image to annotate
        grid_spacing_deg: Spacing between grid lines in degrees
        output_path: Optional path to save annotated image

    Returns:
        Annotated RGB image
    """
    annotated = rgb_image.copy()

    # Draw longitude lines
    for lon in np.arange(0, 360, grid_spacing_deg):
        px, _ = projection.latlon_to_pixel(0, lon)
        px_int = int(round(px))
        if 0 <= px_int < projection.width:
            cv2.line(annotated, (px_int, 0), (px_int, projection.height - 1), (100, 100, 100), 1)

    # Draw latitude lines
    for lat in np.arange(-90, 91, grid_spacing_deg):
        if projection.lat_min <= lat <= projection.lat_max:
            _, py = projection.latlon_to_pixel(lat, projection.lon_min)
            py_int = int(round(py))
            if 0 <= py_int < projection.height:
                cv2.line(annotated, (0, py_int), (projection.width - 1, py_int), (100, 100, 100), 1)

    if output_path:
        cv2.imwrite(str(output_path), annotated)
        print(f"✓ Saved annotated map: {output_path}")

    return annotated
