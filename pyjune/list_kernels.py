"""
Quick script to list all kernel files you have downloaded.
Helps you update the kernel paths in other scripts.
"""

from pathlib import Path


def list_kernels():
    """List all kernel files in the kernels directory."""
    kernel_dir = Path("kernels")

    if not kernel_dir.exists():
        print("No kernels directory found!")
        return

    kernel_types = {
        "lsk": "Leapseconds",
        "pck": "Planetary Constants",
        "fk": "Frames",
        "ik": "Instrument",
        "sclk": "Spacecraft Clock",
        "spk": "Spacecraft Position (MUST cover 2022-056)",
        "ck": "Spacecraft Orientation (MUST cover 2022-056)",
    }

    print("\n" + "=" * 70)
    print("KERNEL INVENTORY")
    print("=" * 70)

    for ktype, description in kernel_types.items():
        kdir = kernel_dir / ktype
        print(f"\n{ktype.upper()} - {description}")
        print("-" * 70)

        if not kdir.exists():
            print("  Directory not found")
            continue

        files = sorted(kdir.glob("*"))
        if not files:
            print("  (empty)")
        else:
            for f in files:
                if f.is_file():
                    size_kb = f.stat().st_size / 1024
                    print(f"  ✓ {f.name} ({size_kb:.1f} KB)")

    print("\n" + "=" * 70)
    print("\nTo use these kernels, update the paths in:")
    print("  - explore_spice.py (line ~23-35)")
    print("  - spice_correction.py (line ~37-52)")
    print("\n")


if __name__ == "__main__":
    list_kernels()
