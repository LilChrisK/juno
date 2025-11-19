"""
Map projection for JunoCam images onto Jupiter's ellipsoid.

This module provides tools to project pushframe images onto Jupiter's surface
using accurate ellipsoidal geometry and SPICE-derived spacecraft state.
"""

import numpy as np
import spiceypy as spice
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import cv2


@dataclass
class SurfacePoint:
    """Represents a point on Jupiter's surface."""

    # Cartesian coordinates in J2000 frame (km)
    position: np.ndarray

    # Planetographic coordinates
    longitude: float  # degrees, 0-360 West
    latitude: float   # degrees, -90 to +90
    altitude: float   # km above ellipsoid

    # Surface properties
    normal: np.ndarray  # Unit normal vector in J2000

    # Source pixel coordinates
    framelet_index: int
    pixel_x: int
    pixel_y: int


class JupiterEllipsoid:
    """
    Jupiter tri-axial ellipsoid model from SPICE.

    Uses SPICE PCK (Planetary Constants Kernel) to get accurate radii.
    Jupiter is an oblate spheroid (rotational ellipsoid) with:
    - Two equal equatorial radii (a = b)
    - One smaller polar radius (c)
    """

    def __init__(self):
        """Initialize ellipsoid from SPICE kernels."""
        # Query Jupiter's radii from SPICE
        # Returns [equatorial_a, equatorial_b, polar_c] in km
        try:
            radii = spice.bodvrd('JUPITER', 'RADII', 3)[1]
            self.a = radii[0]  # Equatorial radius (x-axis)
            self.b = radii[1]  # Equatorial radius (y-axis)
            self.c = radii[2]  # Polar radius (z-axis)

            print(f"Jupiter ellipsoid radii from SPICE:")
            print(f"  Equatorial (a): {self.a:.1f} km")
            print(f"  Equatorial (b): {self.b:.1f} km")
            print(f"  Polar (c): {self.c:.1f} km")
            print(f"  Flattening: {(self.a - self.c) / self.a:.6f}")

        except Exception as e:
            print(f"Warning: Could not query Jupiter radii from SPICE: {e}")
            print("Using default values...")
            self.a = 71492.0  # km
            self.b = 71492.0
            self.c = 66854.0

    def ray_intersection(
        self,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Find intersection of ray with ellipsoid.

        Solves the equation for ray P(t) = origin + t * direction intersecting
        the ellipsoid: (x/a)² + (y/b)² + (z/c)² = 1

        Args:
            ray_origin: Ray starting point (3D vector in J2000, km)
            ray_direction: Ray direction (normalized 3D vector)

        Returns:
            Intersection point (3D vector, km) or None if no intersection
        """
        # Normalize direction
        d = ray_direction / np.linalg.norm(ray_direction)

        # Ray origin
        o = ray_origin

        # Ellipsoid radii
        a, b, c = self.a, self.b, self.c

        # Quadratic coefficients for: At² + Bt + C = 0
        # Substitute ray equation into ellipsoid equation
        A = (d[0]/a)**2 + (d[1]/b)**2 + (d[2]/c)**2
        B = 2 * ((o[0]*d[0])/a**2 + (o[1]*d[1])/b**2 + (o[2]*d[2])/c**2)
        C = (o[0]/a)**2 + (o[1]/b)**2 + (o[2]/c)**2 - 1

        # Discriminant
        discriminant = B**2 - 4*A*C

        if discriminant < 0:
            return None  # No intersection

        # Two solutions
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-B - sqrt_disc) / (2*A)
        t2 = (-B + sqrt_disc) / (2*A)

        # Want the first positive intersection (closest to ray origin)
        t = None
        if t1 > 0:
            t = t1
        elif t2 > 0:
            t = t2
        else:
            return None  # Both intersections behind ray origin

        # Calculate intersection point
        intersection = o + t * d
        return intersection

    def surface_normal(self, point: np.ndarray) -> np.ndarray:
        """
        Calculate outward-pointing normal vector at a surface point.

        For ellipsoid (x/a)² + (y/b)² + (z/c)² = 1,
        the gradient gives the normal: [2x/a², 2y/b², 2z/c²]

        Args:
            point: Point on ellipsoid surface (3D vector, km)

        Returns:
            Unit normal vector (3D, normalized)
        """
        a, b, c = self.a, self.b, self.c

        # Gradient of ellipsoid equation
        normal = np.array([
            2 * point[0] / (a**2),
            2 * point[1] / (b**2),
            2 * point[2] / (c**2)
        ])

        # Normalize
        return normal / np.linalg.norm(normal)

    def body_fixed_to_lonlat(self, point_body_fixed: np.ndarray) -> Tuple[float, float]:
        """
        Conversion from body-fixed Cartesian to lon/lat 
        Computes planetographic longitude and latitude directly from
        IAU_JUPITER body-fixed coordinates using ellipsoid geometry.

        Args:
            point_body_fixed: Point in IAU_JUPITER frame (km)

        Returns:
            (longitude_deg, latitude_deg)
            - Longitude: 0-360° West
            - Latitude: -90° to +90° (planetographic)
        """
        x, y, z = point_body_fixed

        # Longitude: simple atan2 in body-fixed frame
        lon_rad = np.arctan2(y, x)

        # Planetographic latitude for oblate ellipsoid
        # For point on ellipsoid, planetographic lat is angle of surface normal
        # tan(lat) = (z/r_eq) * (a²/c²) where r_eq = sqrt(x² + y²)
        r_eq = np.sqrt(x**2 + y**2)
        lat_rad = np.arctan2(z * self.a**2, r_eq * self.c**2)

        # Convert to degrees
        lon_deg = np.degrees(lon_rad)
        lat_deg = np.degrees(lat_rad)

        # Ensure longitude is in [0, 360) range
        if lon_deg < 0:
            lon_deg += 360.0

        return lon_deg, lat_deg

    def cartesian_to_planetographic(
        self,
        point: np.ndarray,
        et: float
    ) -> Tuple[float, float, float]:
        """
        Convert Cartesian J2000 coordinates to planetographic (lat/lon/alt).

        Uses SPICE to transform from inertial J2000 frame to body-fixed
        IAU_JUPITER frame, then computes geodetic coordinates.

        Args:
            point: Cartesian position in J2000 frame (km)
            et: Ephemeris time (for frame transformation)

        Returns:
            (longitude_deg, latitude_deg, altitude_km)
            - Longitude: 0-360° West (IAU convention for Jupiter)
            - Latitude: -90° to +90° (planetographic)
            - Altitude: height above reference ellipsoid (km)
        """
        # Transform from J2000 to IAU_JUPITER (body-fixed) frame
        rotation = spice.pxform('J2000', 'IAU_JUPITER', et)
        point_body_fixed = rotation @ point

        # Convert to planetographic coordinates
        # recpgr: rectangular to planetographic
        # Returns: (longitude, latitude, altitude)
        lon, lat, alt = spice.recpgr(
            'JUPITER',
            point_body_fixed,
            self.a,  # Equatorial radius
            (self.a - self.c) / self.a  # Flattening factor
        )

        # Convert from radians to degrees
        lon_deg = np.degrees(lon)
        lat_deg = np.degrees(lat)

        return lon_deg, lat_deg, alt

    def planetographic_to_cartesian(
        self,
        lon_deg: float,
        lat_deg: float,
        alt_km: float,
        et: float
    ) -> np.ndarray:
        """
        Convert planetographic coordinates to Cartesian J2000.

        Args:
            lon_deg: Longitude in degrees (0-360 West)
            lat_deg: Latitude in degrees (-90 to +90)
            alt_km: Altitude above ellipsoid (km)
            et: Ephemeris time

        Returns:
            Cartesian position in J2000 frame (km)
        """
        # Convert to radians
        lon_rad = np.radians(lon_deg)
        lat_rad = np.radians(lat_deg)

        # Convert to rectangular body-fixed coordinates
        point_body_fixed = spice.pgrrec(
            'JUPITER',
            lon_rad,
            lat_rad,
            alt_km,
            self.a,
            (self.a - self.c) / self.a
        )

        # Transform from body-fixed to J2000
        rotation = spice.pxform('IAU_JUPITER', 'J2000', et)
        point_j2000 = rotation @ point_body_fixed

        return point_j2000


def get_junocam_fov(camera_id: int = -61500) -> Dict[str, Any]:
    """
    Get JunoCam field-of-view data from SPICE.

    Args:
        camera_id: NAIF ID for JunoCam (-61500)

    Returns:
        Dictionary with FOV parameters
    """
    shape, frame_name, boresight_inst, n, fov_bounds = spice.getfov(camera_id, 16)

    # Calculate FOV extent from boundary vectors
    # fov_bounds are unit direction vectors in instrument frame
    # Convert to angles using atan2
    x_angles = [np.arctan2(fov_bounds[i, 0], fov_bounds[i, 2]) for i in range(n)]
    y_angles = [np.arctan2(fov_bounds[i, 1], fov_bounds[i, 2]) for i in range(n)]

    return {
        'boresight_inst': boresight_inst,
        'fov_x_min_rad': min(x_angles),
        'fov_x_max_rad': max(x_angles),
        'fov_y_min_rad': min(y_angles),
        'fov_y_max_rad': max(y_angles)
    }


class FrameletProjector:
    """
    Projects a single framelet (color band) onto Jupiter's surface.

    Uses SPICE to determine spacecraft state and camera pointing at the
    time of framelet acquisition.
    """

    def __init__(
        self,
        ellipsoid: JupiterEllipsoid,
        et: float,
        framelet_index: int,
        color: str,
        fov_data: Dict[str, Any]
    ):
        """
        Initialize projector for a single framelet.

        Args:
            ellipsoid: Jupiter ellipsoid model
            et: Ephemeris time of framelet exposure
            framelet_index: Frame number
            color: Color name ('red', 'green', or 'blue')
            fov_data: FOV data from get_junocam_fov() (reused across framelets)
        """
        self.ellipsoid = ellipsoid
        self.et = et
        self.framelet_index = framelet_index
        self.color = color

        # Query spacecraft state at framelet time
        state, _ = spice.spkezr('JUNO', et, 'J2000', 'NONE', 'JUPITER')
        self.sc_position = state[:3]  # km
        self.sc_velocity = state[3:]  # km/s

        # Get camera pointing
        self.camera_rotation = spice.pxform('JUNO_JUNOCAM', 'J2000', et)

        # Use provided FOV data
        self.boresight_j2000 = self.camera_rotation @ fov_data['boresight_inst']
        self.fov_x_min_rad = fov_data['fov_x_min_rad']
        self.fov_x_max_rad = fov_data['fov_x_max_rad']
        self.fov_y_min_rad = fov_data['fov_y_min_rad']
        self.fov_y_max_rad = fov_data['fov_y_max_rad']

        # Pre-compute J2000 to IAU_JUPITER rotation for this framelet
        # (same for all pixels since they share ephemeris time)
        self.j2000_to_jupiter_rotation = spice.pxform('J2000', 'IAU_JUPITER', et)

        # Store results
        self.surface_points: List[SurfacePoint] = []

    def project_pixel(
        self,
        pixel_x: int,
        pixel_y: int,
        framelet_height: int = 128,
        framelet_width: int = 1648
    ) -> Optional[SurfacePoint]:
        """
        Project a framelet pixel onto Jupiter's surface.

        Args:
            pixel_x: Horizontal pixel coordinate (0 to framelet_width-1)
            pixel_y: Vertical pixel coordinate (0 to framelet_height-1)
            framelet_height: Height of framelet in pixels
            framelet_width: Width of framelet in pixels

        Returns:
            SurfacePoint if ray hits Jupiter, None otherwise
        """
        # Map pixel coordinates to angles using FOV computed from SPICE kernel
        # Cross-track (X): linear interpolation across detector width
        angle_x_rad = self.fov_x_min_rad + (pixel_x / (framelet_width - 1)) * (self.fov_x_max_rad - self.fov_x_min_rad)

        # Along-track (Y): linear interpolation across detector height
        angle_y_rad = self.fov_y_min_rad + (pixel_y / (framelet_height - 1)) * (self.fov_y_max_rad - self.fov_y_min_rad)

        # Construct ray direction in instrument frame
        # Boresight is +Z, X is cross-track, Y is along-track
        # For a pixel at angles (angle_x, angle_y) from boresight:
        # Use rotation about Y-axis for X angle, then about X-axis for Y angle
        # Or directly: tan(angle_x) = x/z, tan(angle_y) = y/z
        # So: x = tan(angle_x), y = tan(angle_y), z = 1
        ray_inst = np.array([
            np.tan(angle_x_rad),
            np.tan(angle_y_rad),
            1.0
        ])
        ray_inst = ray_inst / np.linalg.norm(ray_inst)

        # Transform to J2000
        ray_j2000 = self.camera_rotation @ ray_inst

        # Intersect with ellipsoid
        intersection = self.ellipsoid.ray_intersection(
            self.sc_position,
            ray_j2000
        )

        if intersection is None:
            return None

        # Calculate surface normal
        normal = self.ellipsoid.surface_normal(intersection)

        # Convert to planetographic using pre-computed rotation
        # Transform from J2000 to IAU_JUPITER (body-fixed) frame
        point_body_fixed = self.j2000_to_jupiter_rotation @ intersection

        # Convert to planetographic coordinates 
        lon_deg, lat_deg = self.ellipsoid.body_fixed_to_lonlat(point_body_fixed)

        return SurfacePoint(
            position=intersection,
            longitude=lon_deg,
            latitude=lat_deg,
            altitude=0.0,  # Not used in orthographic projection, set to 0
            normal=normal,
            framelet_index=self.framelet_index,
            pixel_x=pixel_x,
            pixel_y=pixel_y
        )

    def project_framelet(
        self,
        framelet_data: np.ndarray,
        sample_step: int = 8
    ) -> List[SurfacePoint]:
        """
        Project all pixels in a framelet onto Jupiter's surface.

        Args:
            framelet_data: Image data (height x width array)
            sample_step: Sample every Nth pixel (for speed)

        Returns:
            List of SurfacePoint objects
        """
        height, width = framelet_data.shape
        points = []

        for y in range(0, height, sample_step):
            for x in range(0, width, sample_step):
                point = self.project_pixel(x, y, height, width)
                if point is not None:
                    points.append(point)

        self.surface_points = points
        return points


class OrthographicProjector:
    """
    Creates orthographic map projection of Jupiter from spacecraft viewpoint.

    Orthographic projection shows Jupiter as it appears from the spacecraft,
    with proper foreshortening and limb darkening.
    """

    def __init__(
        self,
        ellipsoid: JupiterEllipsoid,
        center_lon: float,
        center_lat: float,
        et: float,
        map_width: int = 1024,
        map_height: int = 1024,
        scale_km_per_pixel: float = 100.0
    ):
        """
        Initialize orthographic projector.

        Args:
            ellipsoid: Jupiter ellipsoid model
            center_lon: Central longitude (degrees West)
            center_lat: Central latitude (degrees)
            et: Ephemeris time (for coordinate transforms)
            map_width: Output map width in pixels
            map_height: Output map height in pixels
            scale_km_per_pixel: Map scale
        """
        self.ellipsoid = ellipsoid
        self.center_lon = center_lon
        self.center_lat = center_lat
        self.et = et
        self.map_width = map_width
        self.map_height = map_height
        self.scale = scale_km_per_pixel

        # Create output map arrays
        self.map_red = np.zeros((map_height, map_width), dtype=np.float32)
        self.map_green = np.zeros((map_height, map_width), dtype=np.float32)
        self.map_blue = np.zeros((map_height, map_width), dtype=np.float32)
        self.map_counts = np.zeros((map_height, map_width), dtype=np.float32)

    def add_surface_point(
        self,
        point: SurfacePoint,
        pixel_value: float,
        color_channel: str
    ):
        """
        Add a surface point to the map.

        Args:
            point: Surface point with coordinates
            pixel_value: Intensity value from framelet
            color_channel: 'red', 'green', or 'blue'
        """
        # Project lat/lon to orthographic map coordinates
        # Orthographic: x = R * cos(lat) * sin(lon - lon0)
        #               y = R * (cos(lat1) * sin(lat) - sin(lat1) * cos(lat) * cos(lon - lon0))

        lat_rad = np.radians(point.latitude)
        lon_rad = np.radians(point.longitude)
        lat0_rad = np.radians(self.center_lat)
        lon0_rad = np.radians(self.center_lon)

        # Visibility check: only show front hemisphere
        cos_c = (np.sin(lat0_rad) * np.sin(lat_rad) +
                 np.cos(lat0_rad) * np.cos(lat_rad) * np.cos(lon_rad - lon0_rad))

        if cos_c <= 0:
            return  # Point is on back side

        # Orthographic projection
        x_proj = self.ellipsoid.a * np.cos(lat_rad) * np.sin(lon_rad - lon0_rad)
        y_proj = self.ellipsoid.a * (
            np.cos(lat0_rad) * np.sin(lat_rad) -
            np.sin(lat0_rad) * np.cos(lat_rad) * np.cos(lon_rad - lon0_rad)
        )

        # Convert to pixel coordinates
        pixel_x = int(self.map_width / 2 + x_proj / self.scale)
        pixel_y = int(self.map_height / 2 - y_proj / self.scale)

        # Check bounds
        if not (0 <= pixel_x < self.map_width and 0 <= pixel_y < self.map_height):
            return

        # Accumulate into appropriate channel
        if color_channel == 'red':
            self.map_red[pixel_y, pixel_x] += pixel_value
        elif color_channel == 'green':
            self.map_green[pixel_y, pixel_x] += pixel_value
        elif color_channel == 'blue':
            self.map_blue[pixel_y, pixel_x] += pixel_value

        self.map_counts[pixel_y, pixel_x] += 1

    def get_maps(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get final averaged maps.

        Returns:
            (red_map, green_map, blue_map) as float32 arrays
        """
        # Avoid division by zero
        mask = self.map_counts > 0

        red = np.zeros_like(self.map_red)
        green = np.zeros_like(self.map_green)
        blue = np.zeros_like(self.map_blue)

        red[mask] = self.map_red[mask] / self.map_counts[mask]
        green[mask] = self.map_green[mask] / self.map_counts[mask]
        blue[mask] = self.map_blue[mask] / self.map_counts[mask]

        return red, green, blue
