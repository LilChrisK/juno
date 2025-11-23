"""
Test suite for Jupiter coordinate conversions.

Tests round-trip conversions and validates SPICE function usage.
"""

import numpy as np
import spiceypy as spice
from pathlib import Path

from src.spice_correction import SpiceKernelManager
from src.map_projection import JupiterEllipsoid


def test_round_trip_conversions():
    """Test that lat/lon -> cartesian -> lat/lon returns original values."""
    print("\n" + "=" * 70)
    print("TEST: Round-trip coordinate conversions")
    print("=" * 70)

    # Load SPICE kernels
    km = SpiceKernelManager()
    km.load_kernels()

    try:
        # Initialize ellipsoid
        ellipsoid = JupiterEllipsoid()

        # Use a fixed ephemeris time (arbitrary)
        et = spice.str2et("2021-06-08T00:00:00")

        # Test cases: (latitude, longitude) pairs
        # Include edge cases and various locations
        test_cases = [
            (0.0, 0.0, "Equator, Prime Meridian"),
            (0.0, 180.0, "Equator, Antimeridian"),
            (0.0, 90.0, "Equator, 90° West"),
            (0.0, 270.0, "Equator, 270° West (90° East)"),
            (90.0, 0.0, "North Pole"),
            (-90.0, 0.0, "South Pole"),
            (45.0, 0.0, "Mid-latitude North"),
            (-45.0, 0.0, "Mid-latitude South"),
            (30.0, 120.0, "Generic location 1"),
            (-60.0, 240.0, "Generic location 2"),
        ]

        max_error_lat = 0.0
        max_error_lon = 0.0
        all_passed = True

        for lat_in, lon_in, description in test_cases:
            print(f"\n  Testing: {description}")
            print(f"    Input: lat={lat_in:7.3f}°, lon={lon_in:7.3f}°")

            # Convert to radians for SPICE
            lat_rad = np.radians(lat_in)
            lon_rad = np.radians(lon_in)

            # Forward conversion: lat/lon -> cartesian (body-fixed)
            point_body_fixed = spice.pgrrec(
                'JUPITER',
                lon_rad,
                lat_rad,
                0.0,  # altitude
                ellipsoid.equatorial_radius_a,
                (ellipsoid.equatorial_radius_a - ellipsoid.polar_radius) / ellipsoid.equatorial_radius_a
            )

            # Backward conversion: cartesian -> lat/lon
            lon_out, lat_out, alt_out = spice.recpgr(
                'JUPITER',
                point_body_fixed,
                ellipsoid.equatorial_radius_a,
                (ellipsoid.equatorial_radius_a - ellipsoid.polar_radius) / ellipsoid.equatorial_radius_a
            )

            # Convert back to degrees
            lat_out_deg = np.degrees(lat_out)
            lon_out_deg = np.degrees(lon_out)

            # Calculate errors
            error_lat = abs(lat_out_deg - lat_in)
            error_lon = abs(lon_out_deg - lon_in)

            # Handle longitude wrap-around for poles
            # At poles, longitude is undefined, so large errors are expected
            if abs(lat_in) == 90.0:
                # At poles, longitude is meaningless
                error_lon = 0.0
                lon_out_deg = lon_in  # Don't compare
            else:
                # Handle 360° wrap
                if error_lon > 180:
                    error_lon = 360 - error_lon

            max_error_lat = max(max_error_lat, error_lat)
            max_error_lon = max(max_error_lon, error_lon)

            print(f"    Output: lat={lat_out_deg:7.3f}°, lon={lon_out_deg:7.3f}°, alt={alt_out:.3f} km")
            print(f"    Error: lat={error_lat:.6f}°, lon={error_lon:.6f}°")

            # Check tolerance (should be very small, machine precision)
            tolerance = 1e-6  # degrees
            if error_lat > tolerance or error_lon > tolerance:
                print(f"    ✗ FAILED: Error exceeds tolerance ({tolerance}°)")
                all_passed = False
            else:
                print(f"    ✓ PASSED")

        print(f"\n" + "-" * 70)
        print(f"Maximum errors: lat={max_error_lat:.9f}°, lon={max_error_lon:.9f}°")

        if all_passed:
            print("✓ All round-trip tests PASSED")
        else:
            print("✗ Some tests FAILED")

        return all_passed

    finally:
        km.unload_kernels()


def test_frame_transformations():
    """Test J2000 <-> IAU_JUPITER frame transformations."""
    print("\n" + "=" * 70)
    print("TEST: Frame transformations (J2000 <-> IAU_JUPITER)")
    print("=" * 70)

    km = SpiceKernelManager()
    km.load_kernels()

    try:
        ellipsoid = JupiterEllipsoid()
        et = spice.str2et("2021-06-08T00:00:00")

        # Test: transform a point in body-fixed frame to J2000 and back
        # Start with a known point in IAU_JUPITER frame
        lat_deg = 30.0
        lon_deg = 120.0

        print(f"\n  Test location: lat={lat_deg}°, lon={lon_deg}°")

        # Convert to body-fixed cartesian
        lat_rad = np.radians(lat_deg)
        lon_rad = np.radians(lon_deg)
        point_body_fixed = spice.pgrrec(
            'JUPITER',
            lon_rad,
            lat_rad,
            0.0,
            ellipsoid.equatorial_radius_a,
            (ellipsoid.equatorial_radius_a - ellipsoid.polar_radius) / ellipsoid.equatorial_radius_a
        )

        print(f"  Body-fixed (IAU_JUPITER): {point_body_fixed}")

        # Transform to J2000
        rotation_to_j2000 = spice.pxform('IAU_JUPITER', 'J2000', et)
        point_j2000 = rotation_to_j2000 @ point_body_fixed

        print(f"  J2000: {point_j2000}")

        # Transform back to body-fixed
        rotation_to_body = spice.pxform('J2000', 'IAU_JUPITER', et)
        point_body_fixed_2 = rotation_to_body @ point_j2000

        print(f"  Back to body-fixed: {point_body_fixed_2}")

        # Check round-trip error
        error = np.linalg.norm(point_body_fixed - point_body_fixed_2)
        print(f"  Round-trip error: {error:.9f} km")

        tolerance = 1e-6  # km
        if error < tolerance:
            print("  ✓ PASSED")
            return True
        else:
            print(f"  ✗ FAILED: Error {error} exceeds tolerance {tolerance}")
            return False

    finally:
        km.unload_kernels()


def test_altitude_handling():
    """Test that altitude is handled correctly."""
    print("\n" + "=" * 70)
    print("TEST: Altitude handling")
    print("=" * 70)

    km = SpiceKernelManager()
    km.load_kernels()

    try:
        ellipsoid = JupiterEllipsoid()

        lat_deg = 0.0
        lon_deg = 0.0

        # Test different altitudes
        altitudes = [0.0, 100.0, 1000.0, 10000.0]

        all_passed = True
        for alt_in in altitudes:
            print(f"\n  Testing altitude: {alt_in} km")

            lat_rad = np.radians(lat_deg)
            lon_rad = np.radians(lon_deg)

            # Forward
            point_body_fixed = spice.pgrrec(
                'JUPITER',
                lon_rad,
                lat_rad,
                alt_in,
                ellipsoid.equatorial_radius_a,
                (ellipsoid.equatorial_radius_a - ellipsoid.polar_radius) / ellipsoid.equatorial_radius_a
            )

            # Backward
            lon_out, lat_out, alt_out = spice.recpgr(
                'JUPITER',
                point_body_fixed,
                ellipsoid.equatorial_radius_a,
                (ellipsoid.equatorial_radius_a - ellipsoid.polar_radius) / ellipsoid.equatorial_radius_a
            )

            error = abs(alt_out - alt_in)
            print(f"    Input altitude: {alt_in} km")
            print(f"    Output altitude: {alt_out} km")
            print(f"    Error: {error:.9f} km")

            tolerance = 1e-6
            if error < tolerance:
                print(f"    ✓ PASSED")
            else:
                print(f"    ✗ FAILED: Error exceeds tolerance")
                all_passed = False

        return all_passed

    finally:
        km.unload_kernels()


def main():
    """Run all coordinate conversion tests."""
    print("=" * 70)
    print("COORDINATE CONVERSION TEST SUITE")
    print("=" * 70)

    results = []

    # Run tests
    results.append(("Round-trip conversions", test_round_trip_conversions()))
    results.append(("Frame transformations", test_frame_transformations()))
    results.append(("Altitude handling", test_altitude_handling()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
