#!/usr/bin/env python3
"""
Interactive workspace calibration tool.

Usage:
    python3 calibrate_workspace.py [--camera 0] [--image path.jpg]

Workflow:
    1. Captures a frame from the camera (or loads an image file).
    2. Displays the frame and lets you click the 4 corners of the
       10×10 ft workspace in order: top-left, top-right, bottom-right, bottom-left.
    3. Computes a homography mapping pixel → world (metres).
    4. Shows the rectified birds-eye view for verification.
    5. Saves the homography + corners to calibration.yaml.

World frame convention (looking at the overhead image):
    - Origin: CENTER of workspace
    - +X: to the right
    - +Y: upward
    - Coordinates range ±width/2, ±height/2 (e.g. ±1.524 m for 10 ft)
    - This matches the mission CSV coordinate system.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import yaml


WINDOW_NAME = "Calibration — click 4 workspace corners"
CORNER_LABELS = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0,
                    help="Camera index fallback (default 0)")
    ap.add_argument("--config", type=str,
                    default=os.path.join(os.path.dirname(__file__), "config.yaml"),
                    help="Path to config.yaml for camera device path")
    ap.add_argument("--image", type=str, default=None,
                    help="Path to a saved image instead of live camera")
    ap.add_argument("--output", type=str, default=None,
                    help="Output calibration file (default: calibration.yaml next to this script)")
    ap.add_argument("--width-m", type=float, default=3.048,
                    help="Workspace width in metres (default 3.048 = 10 ft)")
    ap.add_argument("--height-m", type=float, default=3.048,
                    help="Workspace height in metres (default 3.048 = 10 ft)")
    return ap.parse_args()


class CornerPicker:
    """Collects 4 mouse-click points on an image."""

    def __init__(self, image):
        self.image = image.copy()
        self.display = image.copy()
        self.points = []

    def _mouse_cb(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            label = CORNER_LABELS[len(self.points) - 1]
            cv2.circle(self.display, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(self.display, label, (x + 12, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if len(self.points) > 1:
                cv2.line(self.display, self.points[-2], self.points[-1],
                         (0, 255, 0), 2)
            if len(self.points) == 4:
                cv2.line(self.display, self.points[-1], self.points[0],
                         (0, 255, 0), 2)
            cv2.imshow(WINDOW_NAME, self.display)

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 960)
        cv2.imshow(WINDOW_NAME, self.display)
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_cb)

        print("\n=== Workspace Calibration ===")
        print("Click the 4 corners of your workspace in this order:")
        for i, label in enumerate(CORNER_LABELS):
            print(f"  {i + 1}. {label}")
        print("Press 'r' to reset, 'q' to quit, Enter to accept.\n")

        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                sys.exit(0)
            elif key == ord("r"):
                self.points.clear()
                self.display = self.image.copy()
                cv2.imshow(WINDOW_NAME, self.display)
                print("Reset — click corners again.")
            elif key in (13, 10) and len(self.points) == 4:
                break

        cv2.destroyAllWindows()
        return np.array(self.points, dtype=np.float32)


def compute_homography(pixel_corners, width_m, height_m):
    """
    Compute homography from pixel corners to world-frame metres.

    pixel_corners order: TL, TR, BR, BL  (as clicked).
    World corners (origin = CENTER, +X right, +Y up):
        TL → (-w/2,  +h/2)
        TR → (+w/2,  +h/2)
        BR → (+w/2,  -h/2)
        BL → (-w/2,  -h/2)
    """
    hw = width_m / 2.0
    hh = height_m / 2.0
    world_corners = np.array([
        [-hw,  hh],   # TL
        [ hw,  hh],   # TR
        [ hw, -hh],   # BR
        [-hw, -hh],   # BL
    ], dtype=np.float32)

    H, status = cv2.findHomography(pixel_corners, world_corners)
    if H is None:
        raise RuntimeError("Homography computation failed — corners may be colinear.")
    return H, world_corners


def show_rectified(image, pixel_corners, width_m, height_m):
    """Warp image into a birds-eye view for visual verification."""
    # Pixels-per-metre for display
    ppm = 200
    dst_w = int(width_m * ppm)
    dst_h = int(height_m * ppm)

    # Pixel corners → display-pixel corners (origin BL → image TL flip)
    dst_corners = np.array([
        [0,     0],        # TL in image
        [dst_w, 0],        # TR
        [dst_w, dst_h],    # BR
        [0,     dst_h],    # BL
    ], dtype=np.float32)

    H_display, _ = cv2.findHomography(pixel_corners, dst_corners)
    warped = cv2.warpPerspective(image, H_display, (dst_w, dst_h))

    # Draw grid every 0.5 m (centered on origin)
    hw = width_m / 2.0
    hh = height_m / 2.0
    for x_m in np.arange(-hw, hw + 0.01, 0.5):
        px = int((x_m + hw) * ppm)
        color = (180, 180, 180) if abs(x_m) < 0.01 else (100, 100, 100)
        cv2.line(warped, (px, 0), (px, dst_h), color, 1 if abs(x_m) > 0.01 else 2)
    for y_m in np.arange(-hh, hh + 0.01, 0.5):
        py = int((hh - y_m) * ppm)  # flip: +Y is up in world, but down in image
        color = (180, 180, 180) if abs(y_m) < 0.01 else (100, 100, 100)
        cv2.line(warped, (0, py), (dst_w, py), color, 1 if abs(y_m) > 0.01 else 2)

    # Label origin (center) and axes
    cx, cy = dst_w // 2, dst_h // 2
    cv2.circle(warped, (cx, cy), 6, (0, 255, 0), -1)
    cv2.putText(warped, "Origin (0,0)", (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.arrowedLine(warped, (cx, cy), (cx + 100, cy), (0, 0, 255), 2)
    cv2.putText(warped, "+X", (cx + 105, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.arrowedLine(warped, (cx, cy), (cx, cy - 100), (0, 255, 0), 2)
    cv2.putText(warped, "+Y", (cx - 25, cy - 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.namedWindow("Rectified birds-eye view", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Rectified birds-eye view", dst_w, dst_h)
    cv2.imshow("Rectified birds-eye view", warped)
    print("Rectified view shown. Press any key to save and exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_calibration(path, H, pixel_corners, width_m, height_m):
    data = {
        "homography": H.tolist(),
        "pixel_corners": {
            "top_left": pixel_corners[0].tolist(),
            "top_right": pixel_corners[1].tolist(),
            "bottom_right": pixel_corners[2].tolist(),
            "bottom_left": pixel_corners[3].tolist(),
        },
        "workspace": {
            "width_m": float(width_m),
            "height_m": float(height_m),
        },
        "world_frame": {
            "origin": "center of workspace",
            "x_axis": "right",
            "y_axis": "up",
            "x_range_m": f"±{float(width_m / 2):.3f}",
            "y_range_m": f"±{float(height_m / 2):.3f}",
        },
    }
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"\nCalibration saved to {path}")


def main():
    args = parse_args()

    # ---- Acquire image ----
    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: could not load image '{args.image}'")
            sys.exit(1)
        print(f"Loaded image: {args.image} ({image.shape[1]}×{image.shape[0]})")
    else:
        # Read camera device from config
        cam_device = args.camera
        if os.path.exists(args.config):
            with open(args.config) as f:
                cfg = yaml.safe_load(f)
            cam_cfg = cfg.get("camera", {})
            cam_device = cam_cfg.get("device", args.camera)

        cap = cv2.VideoCapture(cam_device)
        if not cap.isOpened():
            print(f"Error: could not open camera {cam_device}")
            sys.exit(1)

        if os.path.exists(args.config):
            fourcc = cam_cfg.get("fourcc", None)
            if fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 1280))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 960))
            cap.set(cv2.CAP_PROP_FPS, cam_cfg.get("fps", 30))
        # Auto-focus / auto-exposure warm-up (10 seconds with live preview)
        warmup_sec = 10
        print(f"Camera warming up — auto-focus settling for {warmup_sec}s...")
        cv2.namedWindow("Auto-Focus Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Auto-Focus Preview", 960, 540)
        import time
        t_start = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            elapsed = time.time() - t_start
            remaining = max(0, warmup_sec - elapsed)
            # Draw countdown on preview
            preview = frame.copy()
            cv2.putText(preview, f"Auto-focusing... {remaining:.1f}s",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 255, 255), 3)
            cv2.putText(preview, "Press SPACE to capture early",
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (200, 200, 200), 2)
            cv2.imshow("Auto-Focus Preview", preview)
            key = cv2.waitKey(30) & 0xFF
            if key == ord(" ") or elapsed >= warmup_sec:
                break
        # Capture final frame
        ret, image = cap.read()
        cv2.destroyWindow("Auto-Focus Preview")
        cap.release()
        if not ret:
            print("Error: failed to capture frame")
            sys.exit(1)
        print(f"Captured frame: {image.shape[1]}×{image.shape[0]}")

    # ---- Pick corners ----
    picker = CornerPicker(image)
    pixel_corners = picker.run()
    print(f"\nPixel corners:\n{pixel_corners}")

    # ---- Homography ----
    H, world_corners = compute_homography(pixel_corners, args.width_m, args.height_m)
    print(f"\nHomography matrix:\n{H}")

    # Quick sanity check: transform corners back
    for i, label in enumerate(CORNER_LABELS):
        px = pixel_corners[i].reshape(1, 1, 2)
        world = cv2.perspectiveTransform(px, H).flatten()
        expected = world_corners[i]
        err = np.linalg.norm(world - expected)
        print(f"  {label}: pixel {pixel_corners[i]} → world ({world[0]:.3f}, {world[1]:.3f}) "
              f"[expected ({expected[0]:.3f}, {expected[1]:.3f}), err={err:.4f} m]")

    # ---- Show rectified view ----
    show_rectified(image, pixel_corners, args.width_m, args.height_m)

    # ---- Save ----
    out_path = args.output or os.path.join(os.path.dirname(__file__), "calibration.yaml")
    save_calibration(out_path, H, pixel_corners, args.width_m, args.height_m)


if __name__ == "__main__":
    main()
