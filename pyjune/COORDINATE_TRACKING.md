# Jupiter Storm Tracking with Cylindrical Projections

This document explains how to use the cylindrical projection system to track storms across Jupiter's surface and between perijoves.

## Overview

The system provides:
- **Pixel ↔ Coordinate conversion**: Convert any pixel to Jupiter lat/lon coordinates and vice versa
- **Cylindrical map projections**: Create full maps of Jupiter for analysis
- **Persistent storage**: Save maps with metadata for repeated analysis without reprocessing
- **Multi-perijove tracking**: Foundation for stitching observations across orbits

## Quick Start

### 1. Create a cylindrical projection

```bash
# Full globe, 0.1 deg/pixel resolution
python main.py --mode cylindrical

# Custom region with higher resolution
python main.py --mode cylindrical \
  --cyl-resolution 0.05 \
  --cyl-lat-range -60 60 \
  --cyl-lon-range 0 360
```

### 2. Load and query a saved projection

```python
from pathlib import Path
from cylindrical_projection import CylindricalProjection
from projection_utils import query_pixel_coordinates, query_latlon_pixel

# Load projection
metadata_path = Path("images/processed/cylindrical/JNCE_2021159_34C00080_V01_cylindrical_metadata.json")
projection = CylindricalProjection.load(metadata_path)

# Convert pixel to coordinates
lat, lon = projection.pixel_to_latlon(px=1500, py=800)
print(f"Pixel (1500, 800) → {lat:.2f}°, {lon:.2f}° West")

# Convert coordinates to pixel
px, py = projection.latlon_to_pixel(lat=-22.5, lon=120.0)
print(f"(-22.5°, 120.0°) → Pixel ({px:.1f}, {py:.1f})")
```

### 3. Query projection info

```python
from projection_utils import print_projection_info

print_projection_info(metadata_path)
```

## Coordinate System

- **Latitude**: Planetographic, -90° (South) to +90° (North)
- **Longitude**: System III (1965) West, 0-360°
  - Planetographic longitude is perpendicular to Jupiter's surface
  - System III rotates with Jupiter's magnetic field (9h 55m 29.71s period)
  - "West" means increasing in the direction opposite to rotation

## Projection Details

### Simple Cylindrical (Equirectangular) Projection

Maps latitude and longitude linearly to pixel coordinates:

```
pixel_x = (longitude - lon_min) / resolution
pixel_y = (lat_max - latitude) / resolution
```

**Characteristics:**
- ✓ Simple linear mapping
- ✓ Easy to stitch multiple observations
- ✓ Preserves meridians and parallels
- ✓ Works well for mid-latitudes
- ✗ Area distortion increases toward poles

**When to use:**
- Storm tracking at mid-latitudes (e.g., Great Red Spot at ~22°S)
- Comparing features across perijoves
- Measuring distances along meridians
- Creating global mosaics

## File Structure

After processing, you'll find:

```
images/processed/cylindrical/
└── JNCE_2021159_34C00080_V01/
    ├── JNCE_2021159_34C00080_V01_cylindrical_rgb.png       # RGB composite
    ├── JNCE_2021159_34C00080_V01_cylindrical_red.png       # Red channel
    ├── JNCE_2021159_34C00080_V01_cylindrical_green.png     # Green channel
    ├── JNCE_2021159_34C00080_V01_cylindrical_blue.png      # Blue channel
    └── JNCE_2021159_34C00080_V01_cylindrical_metadata.json # Projection metadata
```

### Metadata Format

```json
{
  "product_id": "JNCE_2021159_34C00080_V01",
  "projection_type": "cylindrical_equirectangular",
  "coordinate_system": "planetographic_system_iii_west",
  "lon_min": 0.0,
  "lon_max": 360.0,
  "lat_min": -90.0,
  "lat_max": 90.0,
  "resolution_deg": 0.1,
  "width": 3600,
  "height": 1800,
  "reference_et": 678901234.56,
  "coverage": {
    "total_pixels": 6480000,
    "valid_pixels": 1234567,
    "coverage_percent": 19.05
  }
}
```

## Examples

### Track the Great Red Spot

The GRS is typically located around:
- Latitude: ~22°S
- Longitude: varies (System II ~250-300° West)

```python
# Find where GRS appears in your map
projection = CylindricalProjection.load(metadata_path)

# Known GRS approximate location (you'll need to verify this for your observation)
grs_lat = -22.0
grs_lon_system_iii = 270.0  # Example - actual position varies with time

# Get pixel location
px, py = projection.latlon_to_pixel(grs_lat, grs_lon_system_iii)
print(f"GRS at pixel ({px:.0f}, {py:.0f})")

# Load the image and check that pixel
import cv2
rgb = cv2.imread("images/processed/cylindrical/..._rgb.png")
# Extract region around (px, py) for analysis
```

### Compare storm locations across perijoves

```python
# Process multiple images
for image_file in ["PJ33_image.png", "PJ34_image.png", "PJ35_image.png"]:
    # Generate projection for each
    # All use same coordinate system, so storms at same lat/lon will align
    pass

# Track storm movement by comparing pixel positions at known coordinates
```

### Create annotated map with grid

```python
from projection_utils import create_annotated_map, load_projection_with_images

projection, rgb, red, green, blue = load_projection_with_images(metadata_path)

# Add 30° grid lines
annotated = create_annotated_map(
    projection,
    rgb,
    grid_spacing_deg=30.0,
    output_path=Path("annotated_map.png")
)
```

## API Reference

### CylindricalProjection

```python
from cylindrical_projection import CylindricalProjection

# Create new projection
proj = CylindricalProjection(
    lon_min=0.0,
    lon_max=360.0,
    lat_min=-90.0,
    lat_max=90.0,
    resolution_deg=0.1
)

# Convert coordinates
lat, lon = proj.pixel_to_latlon(px, py)
px, py = proj.latlon_to_pixel(lat, lon)

# Save/load
proj.save(output_dir, product_id)
proj = CylindricalProjection.load(metadata_path)
```

### Utility Functions

```python
from projection_utils import (
    load_projection_with_images,
    query_pixel_coordinates,
    query_latlon_pixel,
    get_projection_stats,
    print_projection_info,
    create_annotated_map
)
```

## Testing

Run the test suite:

```bash
# Test coordinate conversions
python test_coordinates.py

# Test cylindrical projection with real data
python test_cylindrical.py
```

## Tips for Storm Tracking

1. **Use consistent resolution**: Same deg/pixel for all observations you want to compare

2. **Record System III longitude**: Jupiter's rotation means features move in System II coordinates

3. **Account for temporal changes**: Storms evolve between observations

4. **Check coverage**: Use `get_coverage_map()` to see which pixels have valid data

5. **Save intermediate results**: Generate projections once, analyze repeatedly

6. **Consider latitude range**: If tracking equatorial storms, use narrower lat range for efficiency

## Future Enhancements

Potential additions for advanced tracking:

- **Mosaicking**: Automatically stitch multiple perijoves
- **Feature detection**: Automated storm identification
- **Velocity tracking**: Compute storm drift rates
- **Coordinate registration**: Align observations using feature matching
- **Polar projections**: Better suited for high-latitude features

## References

- [SPICE Planetographic Coordinates](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/pgrrec_c.html)
- [Jupiter System III Longitude](https://pds-atmospheres.nmsu.edu/data_and_services/atmospheres_data/JUNO/juno.html)
- [JunoCam Documentation](https://www.missionjuno.swri.edu/junocam)
