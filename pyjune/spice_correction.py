"""
SPICE-based geometric correction for JunoCam pushframe images.

This module uses SPICE kernels to calculate the spacecraft motion between
filter exposures and correct the resulting image misalignment.
"""

import spiceypy as spice
import numpy as np
from pathlib import Path
import json


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
            self.kernel_dir / "sclk" / "jno_sclkscet_00195.tsc",

            # Spacecraft trajectory (SPK) - replace with file covering your date
            self.kernel_dir / "spk" / "juno_rec_210513_210630_210707.bsp",

            # Spacecraft orientation (CK) - replace with file covering your date
            self.kernel_dir / "ck" / "juno_sc_rec_210606_210612_v01.bc",
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

    def __init__(self, filename, metadata_file=None):
        """
        Initialize from JunoCam filename.

        Filename format: JNCT_YYYYDDD_OOFNNNNN_VXX.EXT
        Example: JNCE_2021159_34C00080_V01-raw.png

        Where:
        - JNC = JunoCam
        - T = product type (E=EDR, R=RDR, M=map)
        - YYYY = year (2021)
        - DDD = day of year (159)
        - OO = orbit number (34)
        - F = filter combination (C)
        - NNNNN = image index (00080)
        - VXX = version (V01)

        Args:
            filename: Path to image file or just the filename
            metadata_file: Optional path to metadata JSON. If None, will search
                          for a JSON file with matching product ID.
        """
        self.filename = Path(filename).name
        self.filepath = Path(filename)
        self.metadata_file = metadata_file
        self.parse_filename()
        self.load_metadata()

    def parse_filename(self):
        """
        Extract metadata from JunoCam filename.

        Format: JNCT_YYYYDDD_OOFNNNNN_VXX
        """
        # Remove extension and any suffix like "-raw"
        base = self.filename.split('.')[0]  # Remove extension
        base = base.split('-')[0]  # Remove suffix like "raw"

        parts = base.split('_')

        if len(parts) != 4:
            raise ValueError(f"Invalid JunoCam filename format: {self.filename}")

        # Parse product type
        product_code = parts[0]  # "JNCE"
        self.product_type = product_code[3] if len(product_code) >= 4 else 'E'

        # Extract year and day of year
        year_doy = parts[1]  # "2021159"
        self.year = int(year_doy[:4])
        self.doy = int(year_doy[4:])

        # Parse image ID: OOFNNNNN
        image_id = parts[2]  # "34C00080"
        self.orbit = int(image_id[:2])  # "34" -> 34
        self.filter_combo = image_id[2]  # "C"
        self.image_index = int(image_id[3:])  # "00080" -> 80

        # Version
        self.version = parts[3]  # "V01"

        # Construct product ID (without extension)
        self.product_id = base

        print(f"Parsed: Year={self.year}, DOY={self.doy}, Orbit={self.orbit}, "
              f"Filter={self.filter_combo}, Index={self.image_index}")

    def load_metadata(self):
        """
        Load metadata from JSON file.

        The metadata contains the actual SPACECRAFT_CLOCK_START_COUNT needed
        for SPICE calculations.

        Strategy:
        1. If metadata_file path provided explicitly, use it
        2. Otherwise, search directory for JSON files
        3. Match by FILE_NAME field in JSON
        """
        if self.metadata_file is None:
            # Search for metadata file in same directory as image
            img_dir = self.filepath.parent

            # Search all JSON files in directory
            json_files = list(img_dir.glob("*.json"))

            if json_files:
                print(f"Searching {len(json_files)} JSON file(s) for matching metadata...")

                for json_file in json_files:
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)

                        # Check if FILE_NAME field matches our image
                        file_name_in_meta = data.get('FILE_NAME', '')

                        if file_name_in_meta == self.filename:
                            self.metadata_file = json_file
                            self.metadata = data  # Store the loaded metadata
                            print(f"✓ Found matching metadata: {json_file.name}")
                            break
                    except (json.JSONDecodeError, KeyError):
                        # Skip invalid JSON files
                        continue

        if self.metadata_file is None:
            print(f"Warning: No metadata file found for {self.filename}")
            print(f"Searched directory: {self.filepath.parent}")
            print("Cannot perform SPICE-based timing calculations without metadata.")
            self.metadata = None
            self.sclk_string = None
            return

        # Load metadata JSON if not already loaded
        if not hasattr(self, 'metadata') or self.metadata is None:
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)

        # Extract SCLK from metadata
        sclk_start = self.metadata.get('SPACECRAFT_CLOCK_START_COUNT')

        if sclk_start:
            # SCLK format in metadata: "676414398:5" (ticks:subseconds)
            # or possibly "partition/ticks:subseconds"
            self.sclk_string = sclk_start
            print(f"Loaded SCLK from metadata: {self.sclk_string}")

            # Also store the image time
            self.image_time = self.metadata.get('IMAGE_TIME')
            print(f"Image time: {self.image_time}")
        else:
            print(f"Warning: No SPACECRAFT_CLOCK_START_COUNT in metadata")
            self.sclk_string = None

    def get_ephemeris_time(self):
        """Convert spacecraft clock to ephemeris time."""
        if self.sclk_string is None:
            # Fallback: use date from filename
            print("No SCLK available, using filename date (less accurate)")
            utc = f"{self.year}-{self.doy:03d}T12:00:00"
            return spice.str2et(utc)

        try:
            et = spice.scs2e(self.JUNO_ID, self.sclk_string)
            return et
        except Exception as e:
            print(f"Error converting SCLK to ET: {e}")
            # Fallback: use IMAGE_TIME from metadata if available
            if hasattr(self, 'image_time') and self.image_time:
                try:
                    return spice.str2et(self.image_time)
                except:
                    pass
            # Final fallback: convert from calendar time
            utc = f"{self.year}-{self.doy:03d}T12:00:00"
            return spice.str2et(utc)

