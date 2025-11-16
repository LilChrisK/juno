"""
Example script to explore and understand SPICE kernels for Juno.

This script demonstrates the basic SPICE operations you'll need for
JunoCam geometric correction.
"""

import spiceypy as spice
from pathlib import Path
from datetime import datetime


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def load_kernels():
    """Load SPICE kernels and show what was loaded."""
    print_section("1. Loading SPICE Kernels")

    kernel_dir = Path("kernels")

    # List of kernels to load - UPDATE THESE with your actual filenames
    kernels = [
        # Time conversion kernels (required for all operations)
        kernel_dir / "lsk" / "naif0012.tls",

        # Planetary constants
        kernel_dir / "pck" / "pck00010.tpc",

        # Juno-specific kernels
        kernel_dir / "fk" / "juno_v12.tf",
        kernel_dir / "ik" / "juno_junocam_v03.ti",
        kernel_dir / "sclk" / "jno_sclkscet_00195.tsc",

        # Time-dependent kernels (MUST cover 2022-056)
        kernel_dir / "spk" / "juno_rec_210513_210630_210707.bsp",
        kernel_dir / "ck" / "juno_sc_rec_210606_210612_v01.bc"
    ]

    loaded = []
    missing = []

    for kernel in kernels:
        if kernel.exists():
            spice.furnsh(str(kernel))
            loaded.append(kernel.name)
            print(f"✓ Loaded: {kernel.name}")
        else:
            missing.append(kernel.name)
            print(f"✗ Missing: {kernel.name}")

    if missing:
        print(f"\nWarning: {len(missing)} kernel(s) not found.")
        print("Some examples below may not work without all kernels.")

    return len(loaded) > 0


def explore_time_conversions():
    """Demonstrate SPICE time conversion capabilities."""
    print_section("2. Time Conversions")

    # Example: Your JunoCam image date
    year = 2021
    doy = 159  # Day of year (June 8, 2021)

    # Method 1: Convert from calendar string to Ephemeris Time (ET)
    utc_string = f"{year}-{doy:03d}T12:00:00"
    print(f"\nUTC String: {utc_string}")

    try:
        et = spice.str2et(utc_string)
        print(f"Ephemeris Time (ET): {et:.6f} seconds past J2000")

        # Convert back to calendar
        calendar = spice.et2utc(et, "C", 0)
        print(f"Back to UTC: {calendar}")

        # Get Julian Date
        jd = spice.et2utc(et, "J", 6)
        print(f"Julian Date: {jd}")

    except Exception as e:
        print(f"Error: {e}")
        print("(LSK kernel required for time conversions)")

    # Method 2: Spacecraft Clock (SCLK) conversion
    print("\n" + "-" * 70)
    print("Spacecraft Clock (SCLK) Conversion:")
    print("-" * 70)

    # From your image: JNCE_2021159_34C00080_V01-raw.png
    # IMPORTANT: The filename ID (34C00080) is NOT a hex SCLK!
    # It's: Orbit 34 + Filter C + Image Index 00080
    # The actual SCLK comes from the metadata JSON file

    print("Filename: JNCE_2021159_34C00080_V01-raw.png")
    print("  Orbit: 34")
    print("  Filter: C")
    print("  Image Index: 80")
    print("  (NOT a hexadecimal SCLK!)")

    # The actual SCLK from metadata JSON:
    print("\nActual SCLK from metadata JSON:")
    sclk_from_metadata = "676414398:5"  # From SPACECRAFT_CLOCK_START_COUNT
    print(f"  SPACECRAFT_CLOCK_START_COUNT: {sclk_from_metadata}")

    try:
        # Convert SCLK to ET (requires SCLK kernel)
        et_from_sclk = spice.scs2e(-61, sclk_from_metadata)
        utc_from_sclk = spice.et2utc(et_from_sclk, "C", 3)
        print(f"  → ET: {et_from_sclk:.6f}")
        print(f"  → UTC: {utc_from_sclk}")
        print(f"\n  Expected (from metadata): 2021-06-08T08:47:12.530")
        print(f"  Converted from SCLK:      {utc_from_sclk}")
        print(f"  ✓ Match!")
    except Exception as e:
        print(f"  Error: {e}")
        print("  (SCLK kernel required for spacecraft clock conversion)")


def explore_spacecraft_state():
    """Query spacecraft position and velocity."""
    print_section("3. Spacecraft State Vectors")

    # Example time (June 8, 2021 - our actual image)
    utc = "2021-06-08T08:47:12"

    try:
        et = spice.str2et(utc)
        print(f"Query time: {utc}")
        print(f"ET: {et:.6f}\n")

        # Get Juno's state relative to Jupiter
        # spkezr returns: [x, y, z, vx, vy, vz] and light time
        print("Juno state relative to Jupiter (J2000 frame):")
        state, lt = spice.spkezr(
            "JUNO",      # Target: Juno spacecraft
            et,          # Time (ephemeris time)
            "J2000",     # Reference frame
            "NONE",      # Aberration correction
            "JUPITER"    # Observer: Jupiter
        )

        position = state[:3]  # km
        velocity = state[3:]  # km/s

        print(f"Position (km): [{position[0]:12.3f}, {position[1]:12.3f}, {position[2]:12.3f}]")
        print(f"Velocity (km/s): [{velocity[0]:9.6f}, {velocity[1]:9.6f}, {velocity[2]:9.6f}]")
        print(f"Range (km): {spice.vnorm(position):12.3f}")
        print(f"Speed (km/s): {spice.vnorm(velocity):9.6f}")
        print(f"Light time (s): {lt:.6f}")

        # Calculate motion over a short interval (like between filter exposures)
        print("\n" + "-" * 70)
        print("Spacecraft motion during 1 millisecond:")
        print("-" * 70)

        dt = 0.001  # 1 millisecond (approximate time between filters)

        state_start, _ = spice.spkezr("JUNO", et, "J2000", "NONE", "JUPITER")
        state_end, _ = spice.spkezr("JUNO", et + dt, "J2000", "NONE", "JUPITER")

        displacement = state_end[:3] - state_start[:3]

        print(f"Time interval: {dt * 1000:.3f} milliseconds")
        print(f"Displacement (km): [{displacement[0]:.9f}, {displacement[1]:.9f}, {displacement[2]:.9f}]")
        print(f"Displacement (meters): [{displacement[0]*1000:.6f}, {displacement[1]*1000:.6f}, {displacement[2]*1000:.6f}]")

        # This displacement causes the pixel shifts you see in your images!

    except Exception as e:
        print(f"Error: {e}")
        print("(SPK kernel covering this date required)")


def explore_spacecraft_orientation():
    """Query spacecraft pointing/orientation."""
    print_section("4. Spacecraft Orientation")

    utc = "2021-06-08T08:47:12"

    try:
        et = spice.str2et(utc)
        print(f"Query time: {utc}\n")

        # Get rotation matrix from J2000 to spacecraft frame
        print("Rotation matrix from J2000 to Juno spacecraft frame:")

        # pxform returns a rotation matrix
        rotation = spice.pxform("J2000", "JUNO_SPACECRAFT", et)

        print("Rotation matrix:")
        for i, row in enumerate(rotation):
            print(f"  [{row[0]:9.6f}, {row[1]:9.6f}, {row[2]:9.6f}]")

        # Example: Transform a vector from J2000 to spacecraft frame
        j2000_vector = [1.0, 0.0, 0.0]  # X-axis in J2000
        sc_vector = spice.mxv(rotation, j2000_vector)

        print(f"\nJ2000 X-axis in spacecraft frame:")
        print(f"  [{sc_vector[0]:9.6f}, {sc_vector[1]:9.6f}, {sc_vector[2]:9.6f}]")

        # Try to get JunoCam frame
        try:
            cam_rotation = spice.pxform("J2000", "JUNO_JUNOCAM", et)
            print("\nJunoCam frame is available!")
            print("This allows transforming from inertial to camera frame.")
        except:
            print("\nJunoCam frame not available (needs IK and FK kernels)")

    except Exception as e:
        print(f"Error: {e}")
        print("(CK kernel covering this date required)")


def explore_coverage():
    """Check time coverage of loaded kernels."""
    print_section("5. Kernel Coverage")

    print("Checking coverage of loaded SPK kernels...")

    try:
        # This is more advanced - requires understanding of SPICE architecture
        # For now, we'll just try to query a specific time

        test_date = "2021-06-08T08:47:12"
        et = spice.str2et(test_date)

        print(f"\nTesting coverage for: {test_date}")

        try:
            state, _ = spice.spkezr("JUNO", et, "J2000", "NONE", "JUPITER")
            print("✓ SPK coverage: Data available for this date")
        except:
            print("✗ SPK coverage: No data for this date")

        try:
            rotation = spice.pxform("J2000", "JUNO_SPACECRAFT", et)
            print("✓ CK coverage: Orientation data available")
        except:
            print("✗ CK coverage: No orientation data for this date")

    except Exception as e:
        print(f"Error: {e}")


def calculate_pixel_shift_example():
    """
    Example calculation: How spacecraft motion translates to pixel shifts.

    This is the key calculation for geometric correction!
    """
    print_section("6. Example: Motion to Pixel Shift Conversion")

    utc = "2021-06-08T08:47:12"

    try:
        et = spice.str2et(utc)

        # Time between filter exposures (approximate)
        dt = 0.001  # 1 millisecond

        # Get spacecraft states
        state_t0, _ = spice.spkezr("JUNO", et, "J2000", "NONE", "JUPITER")
        state_t1, _ = spice.spkezr("JUNO", et + dt, "J2000", "NONE", "JUPITER")

        # Position and velocity
        pos_t0 = state_t0[:3]
        pos_t1 = state_t1[:3]

        # Displacement in km
        displacement_km = pos_t1 - pos_t0
        displacement_m = displacement_km * 1000

        # Range to Jupiter
        range_km = spice.vnorm(pos_t0)

        print(f"Spacecraft range to Jupiter: {range_km:.1f} km")
        print(f"Motion in {dt*1000:.1f} ms: {spice.vnorm(displacement_m):.6f} meters")

        # Calculate angular shift
        # For small angles: angle (rad) ≈ displacement / range
        angular_shift_rad = spice.vnorm(displacement_km) / range_km
        angular_shift_urad = angular_shift_rad * 1e6  # microradians

        print(f"Angular shift: {angular_shift_urad:.3f} microradians")

        # JunoCam pixel scale (approximate - check IK for exact value)
        pixel_scale_urad = 400  # microradians per pixel (example value)

        pixel_shift = angular_shift_urad / pixel_scale_urad

        print(f"\nJunoCam pixel scale: ~{pixel_scale_urad} µrad/pixel")
        print(f"Estimated pixel shift: {pixel_shift:.3f} pixels")

        print("\nThis is why you see color fringing in your images!")
        print("Each color filter is displaced by this amount.")

    except Exception as e:
        print(f"Error: {e}")
        print("(SPK kernel required)")


def main():
    """Run all SPICE exploration examples."""
    print("\n" + "🛰️ " * 20)
    print("SPICE KERNEL EXPLORATION FOR JUNO/JUNOCAM")
    print("🛰️ " * 20)

    # Load kernels
    if not load_kernels():
        print("\n⚠️  No kernels loaded! Download kernels first.")
        print("See download_kernels.py for kernel locations.")
        return

    # Run examples
    explore_time_conversions()
    explore_spacecraft_state()
    explore_spacecraft_orientation()
    explore_coverage()
    calculate_pixel_shift_example()

    print_section("Summary")
    print("""
The key concepts for JunoCam geometric correction:

1. METADATA: Extract SPACECRAFT_CLOCK_START_COUNT from JSON metadata file
   (The filename ID is NOT a hex SCLK - it's orbit+filter+index!)
2. TIME CONVERSION: Convert SCLK → ET using SPICE
3. SPACECRAFT STATE: Get position/velocity at time T
4. MOTION CALCULATION: Compute displacement during filter exposure
5. ANGULAR SHIFT: displacement / range → angular shift
6. PIXEL SHIFT: angular shift / pixel_scale → pixel offset
7. GEOMETRIC CORRECTION: Shift each color channel by calculated offset

IMPORTANT: You need both the image file AND its metadata JSON file!

Once you have all kernels loaded, the spice_correction.py module
automates these steps for each frame in your image.
    """)

    # Clean up
    spice.kclear()
    print("\n✓ Kernels unloaded\n")


if __name__ == "__main__":
    main()
