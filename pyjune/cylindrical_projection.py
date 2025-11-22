"""
Simple cylindrical (equirectangular) map projection for Jupiter.

This module provides a basic cylindrical projection where latitude and longitude
map linearly to pixel coordinates. This is ideal for storm tracking across
multiple perijoves as maps can be easily compared and stitched together.

Projection characteristics:
- Linear mapping: pixel_x ∝ longitude, pixel_y ∝ latitude
- Area distortion increases toward poles
- Preserves vertical meridians and horizontal parallels
- Simple to understand and work with
"""

import numpy as np
import cv2
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from map_projection import JupiterEllipsoid
from coordinates import latlon_to_body_fixed, normalize_longitude


class CylindricalProjection:
    """
    Simple cylindrical (equirectangular) projection of Jupiter.

    Maps lat/lon linearly to pixel coordinates for easy storm tracking.
    """

    def __init__(
        self,
        lon_min: float = 0.0,
        lon_max: float = 360.0,
        lat_min: float = -90.0,
        lat_max: float = 90.0,
        resolution_deg: float = 0.1
    ):
        """
        Initialize cylindrical projection.

        Args:
            lon_min: Minimum longitude (degrees West, 0-360)
            lon_max: Maximum longitude (degrees West, 0-360)
            lat_min: Minimum latitude (degrees, -90 to +90)
            lat_max: Maximum latitude (degrees, -90 to +90)
            resolution_deg: Map resolution in degrees per pixel
        """
        # Store longitude range (don't normalize max if it represents 360)
        self.lon_min = normalize_longitude(lon_min)
        # For lon_max, only normalize if it's not meant to be 360 (full globe)
        if lon_max == 360.0 and lon_min == 0.0:
            self.lon_max = 360.0  # Keep as 360 for full globe
        else:
            self.lon_max = normalize_longitude(lon_max)

        self.lat_min = max(-90.0, min(90.0, lat_min))
        self.lat_max = max(-90.0, min(90.0, lat_max))
        self.resolution_deg = resolution_deg

        # Calculate map dimensions
        # For longitude range calculation, use the original values
        lon_range = lon_max - lon_min
        if lon_range <= 0:  # Handle wrap around 0°
            lon_range += 360.0

        lat_range = lat_max - lat_min

        self.width = int(np.ceil(lon_range / resolution_deg))
        self.height = int(np.ceil(lat_range / resolution_deg))

        # Initialize map arrays
        self.map_red = np.zeros((self.height, self.width), dtype=np.float32)
        self.map_green = np.zeros((self.height, self.width), dtype=np.float32)
        self.map_blue = np.zeros((self.height, self.width), dtype=np.float32)
        self.map_counts = np.zeros((self.height, self.width), dtype=np.float32)

        # Surface grid (computed on demand)
        self.surface_grid = None
        self.reference_et = None

        print(f"Cylindrical projection initialized:")
        print(f"  Longitude range: {self.lon_min:.1f}° to {self.lon_max:.1f}° West")
        print(f"  Latitude range: {self.lat_min:.1f}° to {self.lat_max:.1f}°")
        print(f"  Resolution: {self.resolution_deg:.3f} deg/pixel")
        print(f"  Map size: {self.width} x {self.height} pixels")

    def pixel_to_latlon(self, px: float, py: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to lat/lon.

        Args:
            px: Pixel x-coordinate (0 to width-1)
            py: Pixel y-coordinate (0 to height-1)

        Returns:
            (latitude, longitude) in degrees
        """
        # Linear mapping
        lon = self.lon_min + px * self.resolution_deg
        lat = self.lat_max - py * self.resolution_deg  # Flip y-axis

        lon = normalize_longitude(lon)
        lat = np.clip(lat, -90.0, 90.0)

        return lat, lon

    def latlon_to_pixel(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Convert lat/lon to pixel coordinates.

        Args:
            lat: Latitude in degrees (-90 to +90)
            lon: Longitude in degrees West (0-360)

        Returns:
            (pixel_x, pixel_y) - may be fractional
        """
        lon = normalize_longitude(lon)

        # Handle longitude wrap
        lon_offset = lon - self.lon_min
        if lon_offset < 0:
            lon_offset += 360.0

        px = lon_offset / self.resolution_deg
        py = (self.lat_max - lat) / self.resolution_deg

        return px, py

    def compute_surface_grid(self, ellipsoid: JupiterEllipsoid, et: float) -> np.ndarray:
        """
        Pre-compute 3D surface positions for all map pixels.

        Creates a height × width × 3 array where each pixel corresponds
        to a point on Jupiter's surface in IAU_JUPITER (body-fixed) coordinates.

        Args:
            ellipsoid: Jupiter ellipsoid model
            et: Ephemeris time (stored for reference, not used in computation
                since IAU_JUPITER is body-fixed)

        Returns:
            Surface positions array (height, width, 3) in IAU_JUPITER frame
        """
        print(f"\nComputing surface grid ({self.height} x {self.width})...")

        self.reference_et = et
        surface_positions = np.zeros((self.height, self.width, 3), dtype=np.float32)

        # Generate lat/lon grid
        for i in range(self.height):
            if i % 100 == 0:
                print(f"  Row {i}/{self.height}...")

            for j in range(self.width):
                lat, lon = self.pixel_to_latlon(j, i)

                # Convert to body-fixed Cartesian (IAU_JUPITER frame, altitude = 0)
                # Using body-fixed frame so surface positions are constant
                pos_body_fixed = latlon_to_body_fixed(lat, lon, 0.0, ellipsoid)
                surface_positions[i, j] = pos_body_fixed

        self.surface_grid = surface_positions
        print(f"  Surface grid complete: {self.height * self.width:,} points")

        return surface_positions

    def add_framelet(
        self,
        framelet_data: np.ndarray,
        framelet_cam_position: np.ndarray,
        framelet_cam_orient: np.ndarray,
        ellipsoid,
        sun_position: np.ndarray,
        color_channel: str
    ):
        """
        Add framelet data to the map using backward sampling.

        Simply overwrites with the last observed sample to avoid averaging
        artifacts from overlapping framelets.

        Args:
            framelet_data: Framelet image data (height x width)
            framelet_cam_position: Camera position in IAU_JUPITER frame (km)
            framelet_cam_orient: Camera orientation matrix (JUNO_JUNOCAM -> IAU_JUPITER)
            ellipsoid: JupiterEllipsoid instance for computing surface normals
            sun_position: Sun position in IAU_JUPITER frame (km)
            color_channel: 'red', 'green', or 'blue'
        """
        from framelet_sampling import sample_framelet_at_positions

        if self.surface_grid is None:
            raise RuntimeError("Surface grid not computed. Call compute_surface_grid() first.")

        # Sample framelet at all grid positions
        pixel_values, valid_mask, debug_info = sample_framelet_at_positions(
            self.surface_grid,
            framelet_data,
            framelet_cam_position,
            framelet_cam_orient,
            ellipsoid,
            sun_position,
            color_channel
        )

        # Simply overwrite with new values (last sample wins)
        valid_pixels = valid_mask > 0

        if color_channel == 'red':
            self.map_red[valid_pixels] = pixel_values[valid_pixels]
        elif color_channel == 'green':
            self.map_green[valid_pixels] = pixel_values[valid_pixels]
        elif color_channel == 'blue':
            self.map_blue[valid_pixels] = pixel_values[valid_pixels]

        # Track coverage for statistics
        self.map_counts[valid_pixels] += 1

        return debug_info

    def get_maps(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get final maps (no averaging needed - already using best samples).

        Returns:
            (red_map, green_map, blue_map) as float32 arrays (height x width)
        """
        # Return maps directly - no averaging since we kept only the best sample
        return self.map_red, self.map_green, self.map_blue

    def save(self, output_dir: Path, product_id: str):
        """
        Save projection to disk as PNG images + JSON metadata.

        Args:
            output_dir: Directory to save files
            product_id: Product identifier for filenames
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get final maps
        red, green, blue = self.get_maps()

        # Normalize to 0-255
        def normalize_channel(data):
            if data.max() > 0:
                return (data / data.max() * 255).astype(np.uint8)
            return np.zeros_like(data, dtype=np.uint8)

        red_norm = normalize_channel(red)
        green_norm = normalize_channel(green)
        blue_norm = normalize_channel(blue)

        # Set background (no data) to purple
        no_data_mask = self.map_counts == 0
        red_norm[no_data_mask] = 128
        green_norm[no_data_mask] = 0
        blue_norm[no_data_mask] = 128

        # Create RGB composite (BGR for OpenCV)
        rgb = np.stack([blue_norm, green_norm, red_norm], axis=-1)

        # Save images
        cv2.imwrite(str(output_dir / f"{product_id}_cylindrical_rgb.png"), rgb)
        cv2.imwrite(str(output_dir / f"{product_id}_cylindrical_red.png"), red_norm)
        cv2.imwrite(str(output_dir / f"{product_id}_cylindrical_green.png"), green_norm)
        cv2.imwrite(str(output_dir / f"{product_id}_cylindrical_blue.png"), blue_norm)

        # Save metadata
        metadata = {
            "product_id": product_id,
            "projection_type": "cylindrical_equirectangular",
            "coordinate_system": "planetographic_system_iii_west",
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "resolution_deg": self.resolution_deg,
            "width": self.width,
            "height": self.height,
            "reference_et": self.reference_et,
            "coverage": {
                "total_pixels": int(self.width * self.height),
                "valid_pixels": int(np.sum(self.map_counts > 0)),
                "coverage_percent": float(np.sum(self.map_counts > 0) / (self.width * self.height) * 100)
            }
        }

        with open(output_dir / f"{product_id}_cylindrical_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ Saved cylindrical projection:")
        print(f"  RGB: {output_dir / f'{product_id}_cylindrical_rgb.png'}")
        print(f"  Metadata: {output_dir / f'{product_id}_cylindrical_metadata.json'}")
        print(f"  Coverage: {metadata['coverage']['coverage_percent']:.1f}%")

    @staticmethod
    def load(metadata_path: Path) -> 'CylindricalProjection':
        """
        Load a saved cylindrical projection.

        Args:
            metadata_path: Path to metadata JSON file

        Returns:
            CylindricalProjection instance with loaded parameters
        """
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        projection = CylindricalProjection(
            lon_min=metadata['lon_min'],
            lon_max=metadata['lon_max'],
            lat_min=metadata['lat_min'],
            lat_max=metadata['lat_max'],
            resolution_deg=metadata['resolution_deg']
        )

        projection.reference_et = metadata.get('reference_et')

        print(f"✓ Loaded projection from {metadata_path}")

        return projection
