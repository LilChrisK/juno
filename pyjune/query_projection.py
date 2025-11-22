#!/usr/bin/env python3
"""
Interactive tool to query coordinates in a cylindrical projection.

Usage:
    python query_projection.py <metadata_file>

Examples:
    python query_projection.py images/processed/cylindrical/JNCE_2021159_34C00080_V01_cylindrical_metadata.json
"""

import sys
from pathlib import Path

from cylindrical_projection import CylindricalProjection
from projection_utils import print_projection_info


def main():
    if len(sys.argv) != 2:
        print("Usage: python query_projection.py <metadata_file>")
        print("\nExample:")
        print("  python query_projection.py images/processed/cylindrical/JNCE_2021159_34C00080_V01_cylindrical_metadata.json")
        sys.exit(1)

    metadata_path = Path(sys.argv[1])

    if not metadata_path.exists():
        print(f"Error: File not found: {metadata_path}")
        sys.exit(1)

    # Load projection
    print("Loading projection...")
    projection = CylindricalProjection.load(metadata_path)

    # Print info
    print_projection_info(metadata_path)

    # Interactive query loop
    print("\n" + "=" * 70)
    print("INTERACTIVE COORDINATE QUERY")
    print("=" * 70)
    print("\nCommands:")
    print("  px <x> <y>     - Convert pixel coordinates to lat/lon")
    print("  ll <lat> <lon> - Convert lat/lon to pixel coordinates")
    print("  info           - Show projection info again")
    print("  quit           - Exit")
    print()

    while True:
        try:
            cmd = input(">>> ").strip().lower()

            if not cmd:
                continue

            if cmd == "quit" or cmd == "exit" or cmd == "q":
                print("Goodbye!")
                break

            if cmd == "info":
                print_projection_info(metadata_path)
                continue

            parts = cmd.split()

            if parts[0] == "px" and len(parts) == 3:
                # Pixel to lat/lon
                try:
                    px = float(parts[1])
                    py = float(parts[2])

                    if not (0 <= px < projection.width and 0 <= py < projection.height):
                        print(f"  ✗ Pixel out of bounds. Valid range: x=[0, {projection.width-1}], y=[0, {projection.height-1}]")
                        continue

                    lat, lon = projection.pixel_to_latlon(px, py)
                    print(f"  Pixel ({px:.1f}, {py:.1f}) → Latitude: {lat:.3f}°, Longitude: {lon:.3f}° West")

                except ValueError:
                    print("  ✗ Invalid pixel coordinates. Use: px <x> <y>")

            elif parts[0] == "ll" and len(parts) == 3:
                # Lat/lon to pixel
                try:
                    lat = float(parts[1])
                    lon = float(parts[2])

                    if not (-90 <= lat <= 90):
                        print("  ✗ Latitude out of range. Valid range: -90 to +90")
                        continue

                    px, py = projection.latlon_to_pixel(lat, lon)

                    in_bounds = (0 <= px < projection.width and 0 <= py < projection.height)
                    status = "✓" if in_bounds else "✗ (out of map bounds)"

                    print(f"  {status} Lat {lat:.3f}°, Lon {lon:.3f}° → Pixel ({px:.1f}, {py:.1f})")

                    if not in_bounds:
                        print(f"     Valid pixel range: x=[0, {projection.width-1}], y=[0, {projection.height-1}]")

                except ValueError:
                    print("  ✗ Invalid coordinates. Use: ll <lat> <lon>")

            else:
                print("  ✗ Unknown command. Use: px <x> <y>, ll <lat> <lon>, info, or quit")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"  ✗ Error: {e}")


if __name__ == "__main__":
    main()
