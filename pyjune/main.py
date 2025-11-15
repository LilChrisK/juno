import cv2
import numpy as np
from pathlib import Path
import sys


def main():
    cwd = Path.cwd()
    print(f"Working directory: {cwd}")

    fname = Path("images/raw/JNCE_2022056_40C00036_V01-raw.png")
    raw = cv2.imread(str(fname), cv2.IMREAD_UNCHANGED)
    if raw is None:
        print(f"Could not open raw image: {fname}")
        sys.exit(1)

    height, width = raw.shape[:2]
    rows = height
    print(f"Raw size: {width} x {rows}")

    # Adjust these parameters for your raw file
    bandHeight = 128  # strips height for JunoCam visible filters
    bands = 3  # we are using R, G, B (ignoring methane for now)

    # Offsets (assuming first strip is Blue, then Green, then Red)
    redOffset = 2 * bandHeight
    greenOffset = bandHeight
    blueOffset = 0

    frames = rows // (bandHeight * bands)
    print(f"Frames count: {frames}")

    # Create mosaic per channel
    redMosaic = np.zeros((frames * bandHeight, width), dtype=raw.dtype)
    greenMosaic = np.zeros((frames * bandHeight, width), dtype=raw.dtype)
    blueMosaic = np.zeros((frames * bandHeight, width), dtype=raw.dtype)

    for f in range(1, frames - 1):
        baseRow = f * bandHeight * bands

        # Red
        stripR = raw[baseRow + redOffset : baseRow + redOffset + bandHeight, :]
        redMosaic[
            f * bandHeight + bandHeight : (f + 1) * bandHeight + bandHeight, :
        ] = stripR

        # Green
        stripG = raw[baseRow + greenOffset : baseRow + greenOffset + bandHeight, :]
        greenMosaic[f * bandHeight : (f + 1) * bandHeight, :] = stripG

        # Blue
        stripB = raw[baseRow + blueOffset : baseRow + blueOffset + bandHeight, :]
        blueMosaic[
            f * bandHeight - bandHeight : (f + 1) * bandHeight - bandHeight, :
        ] = stripB

    out_dir = Path("images/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "red_channel.png"), redMosaic)
    cv2.imwrite(str(out_dir / "green_channel.png"), greenMosaic)
    cv2.imwrite(str(out_dir / "blue_channel.png"), blueMosaic)
    print("Single-channel mosaics written.")

    # Convert to 8-bit if raw type is different (just in case)
    red8 = cv2.normalize(redMosaic, None, 0, 255, cv2.NORM_MINMAX)
    green8 = cv2.normalize(greenMosaic, None, 0, 255, cv2.NORM_MINMAX)
    blue8 = cv2.normalize(blueMosaic, None, 0, 255, cv2.NORM_MINMAX)

    red8 = red8.astype(np.uint8)
    green8 = green8.astype(np.uint8)
    blue8 = blue8.astype(np.uint8)

    # Merge into an RGB image (OpenCV is BGR, so put Blue first)
    rgbMosaic = cv2.merge([blue8, green8, red8])

    cv2.imwrite(str(out_dir / "combined_rgb.png"), rgbMosaic)
    print("Combined RGB image written (combined_rgb.png).")


if __name__ == "__main__":
    main()
