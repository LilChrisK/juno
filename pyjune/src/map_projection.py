"""
Map projection for JunoCam images onto Jupiter's ellipsoid.

This module provides the Jupiter ellipsoid model and ray-intersection
calculations used for geometric processing of JunoCam images.
"""

import numpy as np
import spiceypy as spice
from typing import Optional


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
            self.equatorial_radius_a = radii[0]  # Equatorial radius (x-axis)
            self.equatorial_radius_b = radii[1]  # Equatorial radius (y-axis)
            self.polar_radius = radii[2]  # Polar radius (z-axis)

            print(f"Jupiter ellipsoid radii from SPICE:")
            print(f"  Equatorial (a): {self.equatorial_radius_a:.1f} km")
            print(f"  Equatorial (b): {self.equatorial_radius_b:.1f} km")
            print(f"  Polar (c): {self.polar_radius:.1f} km")
            print(f"  Flattening: {(self.equatorial_radius_a - self.polar_radius) / self.equatorial_radius_a:.6f}")

        except Exception as e:
            print(f"Warning: Could not query Jupiter radii from SPICE: {e}")
            print("Using default values...")
            self.equatorial_radius_a = 71492.0  # km
            self.equatorial_radius_b = 71492.0
            self.polar_radius = 66854.0

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
        a, b, c = self.equatorial_radius_a, self.equatorial_radius_b, self.polar_radius

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

    def ray_intersection_vectorized(
        self,
        ray_origin: np.ndarray,
        ray_directions: np.ndarray
    ) -> np.ndarray:
        """
        Find intersections of multiple rays with ellipsoid (vectorized).

        This is the high-performance version that processes all rays at once
        using NumPy broadcasting. Can be 50-200× faster than looping over
        ray_intersection() for large ray batches.

        Solves the equation for rays P(t) = origin + t * direction intersecting
        the ellipsoid: (x/a)² + (y/b)² + (z/c)² = 1

        Args:
            ray_origin: Ray starting point (3D vector in J2000, km)
            ray_directions: Ray directions, shape (..., 3)
                           Can be (H, W, 3) for image rays or (N, 3) for point list

        Returns:
            Intersection points with same shape as ray_directions
            Uses NaN for rays that don't intersect the ellipsoid
        """
        # Store original shape and flatten to (N, 3) for processing
        original_shape = ray_directions.shape
        rays_flat = ray_directions.reshape(-1, 3)
        N = rays_flat.shape[0]

        # Normalize all directions at once
        # Shape: (N, 3)
        norms = np.linalg.norm(rays_flat, axis=1, keepdims=True)
        d = rays_flat / norms

        # Ray origin (broadcast to all rays)
        o = ray_origin  # Shape: (3,)

        # Ellipsoid radii
        a, b, c = self.equatorial_radius_a, self.equatorial_radius_b, self.polar_radius

        # Quadratic coefficients for: At² + Bt + C = 0
        # Substitute ray equation into ellipsoid equation
        # All operations vectorized across N rays
        A = (d[:, 0]/a)**2 + (d[:, 1]/b)**2 + (d[:, 2]/c)**2  # Shape: (N,)
        B = 2 * ((o[0]*d[:, 0])/a**2 + (o[1]*d[:, 1])/b**2 + (o[2]*d[:, 2])/c**2)  # Shape: (N,)
        C = (o[0]/a)**2 + (o[1]/b)**2 + (o[2]/c)**2 - 1  # Scalar, broadcasts

        # Discriminant
        discriminant = B**2 - 4*A*C  # Shape: (N,)

        # Two solutions
        sqrt_disc = np.sqrt(np.maximum(discriminant, 0))  # Clamp negative to 0
        t1 = (-B - sqrt_disc) / (2*A)  # Shape: (N,)
        t2 = (-B + sqrt_disc) / (2*A)  # Shape: (N,)

        # Want the first positive intersection (closest to ray origin)
        # Use vectorized conditional logic
        t = np.where(t1 > 0, t1, t2)  # If t1 > 0 use t1, else use t2

        # Mark invalid intersections as NaN
        # Invalid if: discriminant < 0 OR both t1 and t2 <= 0
        invalid = (discriminant < 0) | ((t1 <= 0) & (t2 <= 0))
        t[invalid] = np.nan

        # Calculate intersection points for all rays
        # Broadcasting: o (3,) + t[:, None] (N, 1) * d (N, 3) = (N, 3)
        intersections = o + t[:, np.newaxis] * d  # Shape: (N, 3)

        # Reshape back to original shape
        return intersections.reshape(original_shape)
