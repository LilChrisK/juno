"""
SPICE-based geometric correction for JunoCam pushframe images.

This module uses SPICE kernels to calculate the spacecraft motion between
filter exposures and correct the resulting image misalignment.
"""

import spiceypy as spice
import numpy as np
from pathlib import Path


class SpiceKernelManager:
    """Manages loading and furnishing SPICE kernels."""

    def __init__(self, kernel_dir="kernels"):
        self.kernel_dir = Path(kernel_dir)
        self.loaded_kernels = []

    def load_kernels(self):
        """Load all required SPICE kernels."""
        # Define kernel paths - adjust these to match your kernel filenames
        kernels = [
            # Leapseconds kernel (for time conversions)
            self.kernel_dir / "lsk" / "naif0012.tls",

            # Planetary constants
            self.kernel_dir / "pck" / "pck00010.tpc",

            # Juno frames kernel
            self.kernel_dir / "fk" / "juno_v12.tf",

            # JunoCam instrument kernel
            self.kernel_dir / "ik" / "juno_junocam_v03.ti",

            # Spacecraft clock kernel
            # Replace with your actual SCLK file
            # Example: self.kernel_dir / "sclk" / "JNO_SCLKSCET.00120.tsc",

            # Spacecraft trajectory (SPK) - replace with file covering your date
            # Example: self.kernel_dir / "spk" / "juno_rec_220101_220401_220405.bsp",

            # Spacecraft orientation (CK) - replace with file covering your date
            # Example: self.kernel_dir / "ck" / "juno_rec_220101_220401_v01.bc",
        ]

        for kernel in kernels:
            if kernel.exists():
                spice.furnsh(str(kernel))
                self.loaded_kernels.append(str(kernel))
                print(f"Loaded: {kernel.name}")
            else:
                print(f"Warning: Kernel not found: {kernel}")

    def unload_kernels(self):
        """Unload all SPICE kernels."""
        spice.kclear()
        self.loaded_kernels = []


class JunoCamImage:
    """Represents a JunoCam image with SPICE-based geometric correction."""

    # JunoCam parameters
    FRAME_ID = -61500  # JunoCam NAIF ID
    JUNO_ID = -61  # Juno spacecraft NAIF ID
    JUPITER_ID = 599  # Jupiter NAIF ID

    # Filter timing (in seconds, approximate values - adjust based on IK)
    FRAME_TRANSFER_TIME = 0.001  # Time between filter exposures

    # Filters are acquired in order: Blue, Green, Red (typically)
    # Each pushframe consists of 3 bands (one per filter)
    FILTER_SEQUENCE = ['BLUE', 'GREEN', 'RED']

    def __init__(self, filename):
        """
        Initialize from JunoCam filename.

        Filename format: JNCE_YYYYDDD_NNNNNNNN_VNN-raw.png
        Example: JNCE_2022056_40C00036_V01-raw.png

        Where:
        - YYYY = year (2022)
        - DDD = day of year (056)
        - NNNNNNNN = image ID (40C00036)
        - VNN = version (V01)
        """
        self.filename = Path(filename).name
        self.parse_filename()

    def parse_filename(self):
        """Extract metadata from JunoCam filename."""
        parts = self.filename.split('_')

        # Extract year and day of year
        year_doy = parts[1]  # "2022056"
        self.year = int(year_doy[:4])
        self.doy = int(year_doy[4:])

        # Extract image ID (hexadecimal spacecraft clock count)
        self.image_id = parts[2]

        # Convert hex image ID to spacecraft clock
        # The image ID is the spacecraft clock count in hexadecimal
        self.sclk_count = int(self.image_id, 16)

        # Convert to string format for SPICE
        self.sclk_string = f"-61/{self.sclk_count}"

        print(f"Parsed: Year={self.year}, DOY={self.doy}, SCLK={self.sclk_string}")

    def get_ephemeris_time(self):
        """Convert spacecraft clock to ephemeris time."""
        try:
            et = spice.scs2e(self.JUNO_ID, self.sclk_string)
            return et
        except Exception as e:
            print(f"Error converting SCLK to ET: {e}")
            # Fallback: convert from calendar time
            utc = f"{self.year}-{self.doy:03d}T00:00:00"
            return spice.str2et(utc)

    def calculate_motion_vector(self, et_start, dt):
        """
        Calculate spacecraft motion vector during time interval.

        Args:
            et_start: Ephemeris time at start of interval
            dt: Time delta in seconds

        Returns:
            Motion vector in JunoCam frame (pixels/second estimated)
        """
        et_end = et_start + dt

        # Get spacecraft position and velocity at both times
        # Relative to Jupiter in J2000 frame
        state_start, _ = spice.spkezr("JUNO", et_start, "J2000", "NONE", "JUPITER")
        state_end, _ = spice.spkezr("JUNO", et_end, "J2000", "NONE", "JUPITER")

        # Extract positions (first 3 components)
        pos_start = state_start[:3]
        pos_end = state_end[:3]

        # Calculate displacement
        displacement = pos_end - pos_start

        # Get spacecraft pointing (C-matrix) to transform to camera frame
        # This requires the CK kernel
        try:
            rotation_matrix = spice.pxform("J2000", "JUNO_JUNOCAM", et_start)
            camera_displacement = rotation_matrix @ displacement
        except Exception as e:
            print(f"Warning: Could not get camera frame transformation: {e}")
            camera_displacement = displacement

        return camera_displacement

    def calculate_pixel_offsets(self, band_height=128, num_frames=None):
        """
        Calculate per-frame pixel offsets for geometric correction.

        Args:
            band_height: Height of each filter band in pixels
            num_frames: Number of pushframes in the image

        Returns:
            Dictionary mapping frame index to (dx, dy) pixel offsets for each filter
        """
        et_base = self.get_ephemeris_time()

        offsets = {}

        if num_frames is None:
            # You'll need to determine this from the image
            num_frames = 30  # example

        # For each pushframe
        for frame_idx in range(num_frames):
            # Calculate time for this frame
            # Each pushframe takes ~band_height * line_time
            # Approximate line time (you should get this from IK kernel)
            line_time = 0.0001  # 100 microseconds per line (example)
            frame_time = frame_idx * band_height * 3 * line_time

            et_frame = et_base + frame_time

            frame_offsets = {}

            # Calculate offset for each filter relative to green (reference)
            for filter_idx, filter_name in enumerate(self.FILTER_SEQUENCE):
                # Time offset from green filter
                if filter_name == 'GREEN':
                    dt = 0.0
                elif filter_name == 'BLUE':
                    dt = -self.FRAME_TRANSFER_TIME
                else:  # RED
                    dt = self.FRAME_TRANSFER_TIME

                # Calculate motion during this time
                motion = self.calculate_motion_vector(et_frame, dt)

                # Convert motion to pixel offsets
                # This requires knowing the camera's pixel scale
                # From JunoCam IK: ~400 microradians/pixel (example value)
                pixel_scale = 400e-6  # radians/pixel

                # Project motion onto image plane
                # This is simplified - real calculation requires full geometry
                # For now, assume small angles
                range_to_jupiter = np.linalg.norm(motion)
                if range_to_jupiter > 0:
                    dx = motion[0] / (range_to_jupiter * pixel_scale)
                    dy = motion[1] / (range_to_jupiter * pixel_scale)
                else:
                    dx, dy = 0, 0

                frame_offsets[filter_name] = (dx, dy)

            offsets[frame_idx] = frame_offsets

        return offsets


def example_usage():
    """Example of how to use SPICE for JunoCam correction."""

    # Initialize SPICE kernels
    km = SpiceKernelManager()
    km.load_kernels()

    try:
        # Parse image metadata
        img = JunoCamImage("JNCE_2022056_40C00036_V01-raw.png")

        # Get ephemeris time
        et = img.get_ephemeris_time()
        print(f"Ephemeris time: {et}")

        # Calculate pixel offsets for correction
        offsets = img.calculate_pixel_offsets(band_height=128, num_frames=30)

        # Print example offsets
        print("\nExample offsets for frame 15:")
        print(offsets[15])

        # These offsets can then be used to shift each color channel
        # in your image processing code

    finally:
        # Clean up
        km.unload_kernels()


if __name__ == "__main__":
    example_usage()
