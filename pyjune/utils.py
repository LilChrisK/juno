import cv2
import numpy as np


def create_debug_visualization(fname, output_path):
    """
    Create a debug image showing frame and band structure.

    Args:
        fname: Path to raw JunoCam image
        output_path: Where to save the debug image
    """
    # Load raw image
    raw = cv2.imread(str(fname), cv2.IMREAD_UNCHANGED)
    if raw is None:
        print(f"Could not open raw image: {fname}")
        return

    height, width = raw.shape[:2]

    # Convert to BGR for colored overlays
    if len(raw.shape) == 2:  # Grayscale
        debug_img = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    else:
        debug_img = raw.copy()

    # Normalize to 8-bit for visualization
    debug_img = cv2.normalize(debug_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # JunoCam parameters
    band_height = 128
    bands = 3
    frame_height = band_height * bands
    frames = height // frame_height

    color_map = {0: 'BLUE', 1: 'GREEN', 2: 'RED'}

    # Draw frame separators (red) and band separators (blue)
    for f in range(frames + 1):
        y = f * frame_height
        if y < height:
            # Red line for frame boundary
            cv2.line(debug_img, (0, y), (width, y), (0, 0, 255), 3)

    for f in range(frames):
        for b in range(1, bands):  # Don't redraw frame boundaries
            y = f * frame_height + b * band_height
            if y < height:
                # Blue line for band boundary
                cv2.line(debug_img, (0, y), (width, y), (255, 0, 0), 2)

    # Label each framelet
    for f in range(frames):
        for color_idx in range(bands):
            y_center = f * frame_height + color_idx * band_height + band_height // 2

            # Create label
            label = f"F{f} {color_map[color_idx]}"

            # Put text with background for readability
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Draw black background rectangle
            cv2.rectangle(debug_img,
                         (10, y_center - text_height - 5),
                         (10 + text_width + 10, y_center + 5),
                         (0, 0, 0), -1)

            # Draw text
            cv2.putText(debug_img, label, (15, y_center),
                       font, font_scale, (255, 255, 255), thickness)

    # Save debug image
    cv2.imwrite(str(output_path), debug_img)
    print(f"Debug visualization saved to {output_path}")

