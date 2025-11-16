# PyJune - JunoCam Image Processing with SPICE

Tools for processing NASA Juno JunoCam images with SPICE-based geometric correction.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install spiceypy opencv-python numpy scipy
   ```

2. **Download SPICE kernels** (see `SPICE_README.md`)

3. **Verify setup:**
   ```bash
   python test_spice_setup.py
   ```

4. **Process images:**
   ```bash
   python main_with_spice.py
   ```

## Project Files

### Core Processing
- **`main.py`** - Original image processing (basic RGB channel separation)
- **`main_with_spice.py`** - SPICE-corrected image processing
- **`spice_correction.py`** - SPICE geometric correction module

### Learning & Testing
- **`explore_spice.py`** - Interactive SPICE tutorial with examples
- **`test_spice_setup.py`** - Verify your SPICE setup before processing

### Utilities
- **`download_kernels.py`** - Helper for downloading SPICE kernels
- **`list_kernels.py`** - Show what kernels you have

### Documentation
- **`SPICE_README.md`** - Complete SPICE setup and usage guide
- **`FILENAME_FORMAT_DISCOVERY.md`** - How we figured out JunoCam filename format

## File Requirements

To process a JunoCam image, you need:

1. **Image file**: `JNCE_2021159_34C00080_V01-raw.png`
2. **Metadata JSON**: Contains `SPACECRAFT_CLOCK_START_COUNT` (required for SPICE timing)
3. **SPICE kernels**: LSK, PCK, FK, IK, SCLK, SPK, CK covering your observation date

## JunoCam Filename Format

`JNCT_YYYYDDD_OOFNNNNN_VXX`

Example: `JNCE_2021159_34C00080_V01-raw.png`
- **JNC**: JunoCam
- **E**: EDR (Experiment Data Record)
- **2021159**: Year 2021, day 159 (June 8)
- **34**: Orbit 34 (Perijove 34)
- **C**: Filter combination
- **00080**: Image index 80
- **V01**: Version 01

## Geometric Correction

JunoCam uses pushframe imaging - each color (R, G, B) is captured sequentially. Between exposures, the spacecraft moves, causing color misalignment.

SPICE provides:
- Spacecraft position and velocity
- Spacecraft orientation
- Precise timing from spacecraft clock

Using this data, we calculate pixel shifts between color channels and correct the alignment.

## Workflow

1. Parse JunoCam filename → extract date, orbit
2. Load metadata JSON → get `SPACECRAFT_CLOCK_START_COUNT`
3. Convert SCLK → Ephemeris Time using SPICE
4. Query spacecraft state at each frame time
5. Calculate motion between R/G/B exposures
6. Convert motion → pixel offsets
7. Apply geometric correction to each channel
8. Merge corrected channels → aligned RGB image

## References

- [JunoCam Mission Page](https://www.missionjuno.swri.edu/junocam)
- [NAIF SPICE Toolkit](https://naif.jpl.nasa.gov/naif/)
- [SpiceyPy Documentation](https://spiceypy.readthedocs.io/)
