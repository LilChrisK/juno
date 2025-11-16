"""
Script to help download and manage SPICE kernels for Juno JunoCam processing.

Required kernels from: https://naif.jpl.nasa.gov/pub/naif/pds/data/jno-j_e_ss-spice-6-v1.0/jnosp_1000/data/

For your image: JNCE_2022056_40C00036_V01-raw.png
- Date: 2022, day 056 (February 25, 2022)

Download the following kernels to the respective directories:

1. LSK (Leapseconds) - kernels/lsk/
   - Latest naif0012.tls from: .../lsk/

2. PCK (Planetary Constants) - kernels/pck/
   - pck00011.tpc from: .../pck/

3. FK (Frames) - kernels/fk/
   - juno_v12.tf (or latest) from: .../fk/

4. IK (Instrument) - kernels/ik/
   - juno_junocam_v03.ti (or latest) from: .../ik/

5. SCLK (Spacecraft Clock) - kernels/sclk/
   - JNO_SCLKSCET.00xxx.tsc (use latest or one covering 2022) from: .../sclk/

6. SPK (Spacecraft trajectory) - kernels/spk/
   - reconstructed SPK covering Feb 2022 from: .../spk/
   - Look for files like juno_rec_*.bsp or juno_struct_*.bsp

7. CK (Spacecraft orientation) - kernels/ck/
   - reconstructed CK covering Feb 2022 from: .../ck/
   - Look for files like juno_rec_*.bc

Note: You'll need to find the specific SPK and CK files that cover your date range.
"""

import urllib.request
from pathlib import Path

# Base URL for Juno kernels (PDS archive)
BASE_URL = "https://naif.jpl.nasa.gov/pub/naif/pds/data/jno-j_e_ss-spice-6-v1.0/jnosp_1000/data/"

# Kernels that don't change (you can auto-download these)
STATIC_KERNELS = {
    "lsk/naif0012.tls": "kernels/lsk/naif0012.tls",
    "pck/pck00011.tpc": "kernels/pck/pck00011.tpc",
}

def download_kernel(url, dest):
    """Download a kernel file if it doesn't exist."""
    dest_path = Path(dest)
    if dest_path.exists():
        print(f"Already exists: {dest}")
        return

    print(f"Downloading {url} -> {dest}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"Downloaded: {dest}")

def main():
    print("Downloading static SPICE kernels...")
    for kernel_path, local_path in STATIC_KERNELS.items():
        url = BASE_URL + kernel_path
        try:
            download_kernel(url, local_path)
        except Exception as e:
            print(f"Error downloading {url}: {e}")

    print("\n" + "="*60)
    print("Manual downloads needed:")
    print("="*60)
    print("\nVisit these URLs to find kernels covering Feb 2022:")
    print(f"- FK: {BASE_URL}fk/")
    print(f"- IK: {BASE_URL}ik/")
    print(f"- SCLK: {BASE_URL}sclk/")
    print(f"- SPK: {BASE_URL}spk/")
    print(f"- CK: {BASE_URL}ck/")
    print("\nFor image date 2022-056 (Feb 25, 2022), you need:")
    print("- SPK file covering this date (look for juno_rec_YYMMDD_YYMMDD_*.bsp)")
    print("- CK file covering this date (look for juno_rec_YYMMDD_YYMMDD_*.bc)")
    print("- Latest SCLK file created after Feb 2022")
    print("\nNote: In PDS archive, check subdirectories within each kernel type folder.")

if __name__ == "__main__":
    main()
