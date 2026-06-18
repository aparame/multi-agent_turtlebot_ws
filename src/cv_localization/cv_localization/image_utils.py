"""Image helpers shared by ROS nodes and unit tests."""

import cv2
import numpy as np


def crop_resize_workspace_frame(frame, workspace_pixel_corners, output_size_px):
    """Perspective-crop the calibrated workspace and resize to a square image."""
    output_size_px = max(1, int(output_size_px))
    src = np.asarray(workspace_pixel_corners, dtype=np.float32)
    if src.shape != (4, 2):
        raise ValueError("workspace_pixel_corners must contain four (x, y) points")
    dst = np.array([
        [0.0, 0.0],
        [float(output_size_px - 1), 0.0],
        [float(output_size_px - 1), float(output_size_px - 1)],
        [0.0, float(output_size_px - 1)],
    ], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        frame,
        transform,
        (output_size_px, output_size_px),
        flags=cv2.INTER_AREA,
    )
