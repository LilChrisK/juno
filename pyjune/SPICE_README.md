# JunoCam SPICE-Based Geometric Correction

This implementation uses NASA's SPICE toolkit to correct geometric distortions in JunoCam pushframe images caused by spacecraft motion.

## How It Works

JunoCam captures images using a pushframe technique where each frame consists of 3 separate exposures through different color filters (Blue, Green, Red). Between each filter exposure, the spacecraft moves, causing the three color channels to be misaligned.

SPICE kernels provide:
- Spacecraft position and velocity
- Spacecraft orientation
- Instrument timing and geometry
- Planetary body parameters

Using this data, we calculate the exact pixel shift between filter exposures and correct the alignment.

## Setup

### 1. Install Dependencies

```bash
pip install spiceypy opencv-python numpy scipy
```

### 2. Download SPICE Kernels

Download the following kernels from https://naif.jpl.nasa.gov/pub/naif/JUNO/kernels/

Place them in the `kernels/` directory structure:

```
kernels/
├── lsk/naif0012.tls
├── pck/pck00010.tpc
├── fk/juno_v12.tf
├── ik/juno_junocam_v03.ti
├── sclk/JNO_SCLKSCET.00xxx.tsc
├── spk/<reconstruction SPK covering your date>
└── ck/<reconstruction CK covering your date>
```

**Important**: You need to find the specific SPK and CK files that cover your image date.

For image `JNCE_2022056_40C00036_V01-raw.png` (Feb 25, 2022), look for:
- SPK files like: `juno_rec_220101_220401_*.bsp`
- CK files like: `juno_rec_220101_220401_*.bc`

### 3. Update Kernel Paths

Edit `spice_correction.py` line ~37-52 to match your actual kernel filenames:

```python
# Replace these with your actual kernel files
self.kernel_dir / "sclk" / "JNO_SCLKSCET.00120.tsc",
self.kernel_dir / "spk" / "juno_rec_220101_220401_220405.bsp",
self.kernel_dir / "ck" / "juno_rec_220101_220401_v01.bc",
```

## Usage

### Basic Usage

```bash
python main_with_spice.py
```

This will:
1. Load SPICE kernels
2. Parse the image timestamp
3. Calculate per-frame geometric corrections
4. Apply corrections to each color channel
5. Save corrected images to `images/processed/`

### Advanced Usage

```python
from spice_correction import SpiceKernelManager, JunoCamImage

# Initialize SPICE
km = SpiceKernelManager()
km.load_kernels()

# Parse image
img = JunoCamImage("JNCE_2022056_40C00036_V01-raw.png")

# Get corrections
offsets = img.calculate_pixel_offsets(band_height=128, num_frames=30)

# offsets[frame_idx]['RED'] -> (dx, dy) in pixels
# offsets[frame_idx]['GREEN'] -> (dx, dy) in pixels
# offsets[frame_idx]['BLUE'] -> (dx, dy) in pixels

km.unload_kernels()
```

## Output

The script produces:
- `red_channel_spice.png` - Red filter mosaic (corrected)
- `green_channel_spice.png` - Green filter mosaic (corrected)
- `blue_channel_spice.png` - Blue filter mosaic (corrected)
- `combined_rgb_spice.png` - RGB composite (corrected)

Compare these to the uncorrected versions to see the improvement in color alignment.

## Parameters to Tune

### In `spice_correction.py`:

- **FRAME_TRANSFER_TIME** (line ~23): Time between filter exposures (~0.001s)
- **pixel_scale** (line ~156): Camera pixel scale in radians/pixel (~400e-6)
- **line_time** (line ~182): Time to acquire one line of pixels (~100e-6s)

These values should ideally come from the JunoCam IK (instrument kernel), but may need empirical tuning.

### In `main_with_spice.py`:

- **bandHeight** (line ~62): Height of each filter band (128 pixels for JunoCam)
- **bands** (line ~63): Number of color filters (3 for RGB)

## Troubleshooting

### "Kernel not found" errors
- Verify kernel paths in `spice_correction.py`
- Ensure kernels are downloaded to correct directories

### "SPICE error" messages
- Check that SPK/CK kernels cover your image date
- Verify SCLK kernel is compatible with your image

### Minimal/no correction applied
- Tune timing parameters (FRAME_TRANSFER_TIME, line_time)
- Verify spacecraft was moving significantly during acquisition
- Check that CK kernel has orientation data for your time

### Strange offsets
- Verify pixel_scale value matches JunoCam specs
- Check coordinate frame transformations
- Ensure you're using reconstructed (not predicted) kernels

## Next Steps

After geometric correction, you can:
1. Reproject onto Jupiter ellipsoid
2. Apply photometric corrections
3. Color balance and enhance
4. Map project to cylindrical/orthographic views

## References

- [NAIF SPICE Toolkit](https://naif.jpl.nasa.gov/naif/)
- [SpiceyPy Documentation](https://spiceypy.readthedocs.io/)
- [JunoCam Information](https://www.missionjuno.swri.edu/junocam)
- [Processing JunoCam Images](https://www.missionjuno.swri.edu/junocam/processing)
