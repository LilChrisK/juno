"""
Jupiter coordinate conversion utilities.

Provides conversions between planetographic (lat/lon/alt) and Cartesian
coordinates in both body-fixed (IAU_JUPITER) and inertial (J2000) frames.

All functions use SPICE for accurate conversions that account for Jupiter's
oblate ellipsoid shape.
"""

import numpy as np
import spiceypy as spice
from typing import Tuple

from src.map_projection import JupiterEllipsoid


def latlon_to_body_fixed(
    lat_deg: float,
    lon_deg: float,
    alt_km: float,
    ellipsoid: JupiterEllipsoid
) -> np.ndarray:
    """
    Convert planetographic coordinates to body-fixed Cartesian (IAU_JUPITER frame).

    Args:
        lat_deg: Planetographic latitude in degrees (-90 to +90)
        lon_deg: System III (1965) West longitude in degrees (0-360)
        alt_km: Altitude above reference ellipsoid in km
        ellipsoid: Jupiter ellipsoid model

    Returns:
        Position in IAU_JUPITER frame (km) as 3D numpy array
    """
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)

    flattening = (ellipsoid.equatorial_radius_a - ellipsoid.polar_radius) / ellipsoid.equatorial_radius_a

    point_body_fixed = spice.pgrrec(
        'JUPITER',
        lon_rad,
        lat_rad,
        alt_km,
        ellipsoid.equatorial_radius_a,
        flattening
    )

    return point_body_fixed


def body_fixed_to_latlon(
    point_body_fixed: np.ndarray,
    ellipsoid: JupiterEllipsoid
) -> Tuple[float, float, float]:
    """
    Convert body-fixed Cartesian to planetographic coordinates.

    Args:
        point_body_fixed: Position in IAU_JUPITER frame (km)
        ellipsoid: Jupiter ellipsoid model

    Returns:
        Tuple of (lat_deg, lon_deg, alt_km):
            - lat_deg: Planetographic latitude (-90 to +90)
            - lon_deg: System III West longitude (0-360)
            - alt_km: Altitude above reference ellipsoid
    """
    flattening = (ellipsoid.equatorial_radius_a - ellipsoid.polar_radius) / ellipsoid.equatorial_radius_a

    lon_rad, lat_rad, alt_km = spice.recpgr(
        'JUPITER',
        point_body_fixed,
        ellipsoid.equatorial_radius_a,
        flattening
    )

    lat_deg = np.degrees(lat_rad)
    lon_deg = np.degrees(lon_rad)

    return lat_deg, lon_deg, alt_km


def latlon_to_j2000(
    lat_deg: float,
    lon_deg: float,
    alt_km: float,
    ellipsoid: JupiterEllipsoid,
    et: float
) -> np.ndarray:
    """
    Convert planetographic coordinates to J2000 inertial frame.

    Args:
        lat_deg: Planetographic latitude in degrees (-90 to +90)
        lon_deg: System III West longitude in degrees (0-360)
        alt_km: Altitude above reference ellipsoid in km
        ellipsoid: Jupiter ellipsoid model
        et: Ephemeris time (seconds past J2000)

    Returns:
        Position in J2000 frame (km) as 3D numpy array
    """
    # First convert to body-fixed
    point_body_fixed = latlon_to_body_fixed(lat_deg, lon_deg, alt_km, ellipsoid)

    # Then transform to J2000
    rotation = spice.pxform('IAU_JUPITER', 'J2000', et)
    point_j2000 = rotation @ point_body_fixed

    return point_j2000


def j2000_to_latlon(
    point_j2000: np.ndarray,
    ellipsoid: JupiterEllipsoid,
    et: float
) -> Tuple[float, float, float]:
    """
    Convert J2000 inertial coordinates to planetographic.

    Args:
        point_j2000: Position in J2000 frame (km)
        ellipsoid: Jupiter ellipsoid model
        et: Ephemeris time (seconds past J2000)

    Returns:
        Tuple of (lat_deg, lon_deg, alt_km):
            - lat_deg: Planetographic latitude (-90 to +90)
            - lon_deg: System III West longitude (0-360)
            - alt_km: Altitude above reference ellipsoid
    """
    # Transform from J2000 to body-fixed
    rotation = spice.pxform('J2000', 'IAU_JUPITER', et)
    point_body_fixed = rotation @ point_j2000

    # Then convert to lat/lon
    return body_fixed_to_latlon(point_body_fixed, ellipsoid)


def normalize_longitude(lon_deg: float) -> float:
    """
    Normalize longitude to [0, 360) range.

    Args:
        lon_deg: Longitude in degrees (any range)

    Returns:
        Longitude normalized to [0, 360) degrees
    """
    lon_normalized = np.fmod(lon_deg, 360.0)
    if lon_normalized < 0:
        lon_normalized += 360.0
    return lon_normalized
