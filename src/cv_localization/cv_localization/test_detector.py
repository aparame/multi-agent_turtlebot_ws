#!/usr/bin/env python3
"""
Standalone test: run the marker-free detector on a live camera feed.

Usage:
    python3 test_detector.py                          # live camera
    python3 test_detector.py --image snapshot.jpg     # single image

Workflow:
    1. Press 'b' to capture background (with robots OUT of workspace).
    2. Place robots in workspace.
    3. Press 'i' to initialize identities (assigns tb_1, tb_2, tb_3
       based on left-to-right order; or provide a mission CSV).
    4. Tracking begins automatically.

Other keys:
    'b' — recapture background
    'i' — re-initialize identities
    'm' — toggle detection mask overlay
    's' — save snapshot
    'q' — quit
"""

import argparse
import math
import os
import sys
import time

import cv2
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from detector import RobotDetector


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--image", type=str, default=None,
                    help="Test on a single image")
    ap.add_argument("--background", type=str, default=None,
                    help="Path to a saved background image")
    ap.add_argument("--config", type=str,
                    default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("--calibration", type=str,
                    default=os.path.join(os.path.dirname(__file__), "calibration.yaml"))
    return ap.parse_args()


def open_camera(config_path, fallback_index=0):
    """Open camera using config.yaml device path."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cam_cfg = cfg.get("camera", {})
    cam_device = cam_cfg.get("device", fallback_index)

    cap = cv2.VideoCapture(cam_device)
    if not cap.isOpened():
        print(f"Error: cannot open camera {cam_device}")
        sys.exit(1)

    fourcc = cam_cfg.get("fourcc", None)
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 960))
    cap.set(cv2.CAP_PROP_FPS, cam_cfg.get("fps", 30))
    return cap


def main():
    args = parse_args()

    if not os.path.exists(args.calibration):
        print(f"Error: calibration file not found: {args.calibration}")
        print("Run calibrate_workspace.py first!")
        sys.exit(1)

    detector = RobotDetector(args.config, args.calibration)
    print(f"Detector initialized.  Num robots: {detector.num_robots}")
    print(f"Background subtraction: {'loaded' if detector.has_background() else 'NOT SET'}")

    # Load pre-saved background if provided
    if args.background:
        bg_img = cv2.imread(args.background)
        if bg_img is not None:
            detector.set_background(bg_img)
            print(f"Background loaded from {args.background}")
        else:
            print(f"Warning: could not load background {args.background}")

    if args.image:
        # --- Static image mode ---
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Error: cannot load {args.image}")
            sys.exit(1)

        if not detector.has_background():
            print("Warning: no background image — using dark-threshold only")

        poses = detector.detect(frame)
        vis = detector.draw_detections(frame, poses, show_mask=True)

        print(f"\nDetected {len(poses)} robots:")
        for rid, p in sorted(poses.items()):
            print(f"  tb_{rid}: x={p.x:.3f} y={p.y:.3f} [{p.source}]")

        cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detections", 1280, 960)
        cv2.imshow("Detections", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # --- Live camera mode ---
    cap = open_camera(args.config, args.camera)

    cv2.namedWindow("CV Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CV Tracker", 1280, 720)

    fps_history = []
    show_mask = False
    identities_set = False

    print("\n" + "=" * 60)
    print("  MARKER-FREE TURTLEBOT TRACKER — TEST MODE")
    print("=" * 60)
    print("  1. Press 'b' to capture BACKGROUND (robots OUT of view)")
    print("  2. Place robots in workspace")
    print("  3. Press 'i' to INITIALIZE identities")
    print("  4. Tracking runs automatically")
    print("  'm' = toggle mask | 's' = snapshot | 'q' = quit")
    print("=" * 60 + "\n")

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        poses = detector.detect(frame) if identities_set else {}
        vis = detector.draw_detections(frame, poses, show_mask=show_mask)

        # Status bar
        status = "READY"
        if not detector.has_background():
            status = "Press 'b' to capture BACKGROUND"
        elif not identities_set:
            status = "Place robots, press 'i' to INITIALIZE"
        else:
            status = f"TRACKING {len(poses)}/{detector.num_robots} robots"

        cv2.putText(vis, status, (10, vis.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # FPS
        dt = time.time() - t0
        fps_history.append(dt)
        if len(fps_history) > 30:
            fps_history.pop(0)
        fps = 1.0 / (sum(fps_history) / len(fps_history)) if fps_history else 0
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("CV Tracker", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("b"):
            # Capture background
            detector.set_background(frame)
            bg_path = os.path.join(os.path.dirname(__file__), "background.jpg")
            cv2.imwrite(bg_path, frame)
            print(f"Background captured and saved to {bg_path}")

        elif key == ord("i"):
            # Initialize identities from current blob positions
            if not detector.has_background():
                print("Error: capture background first (press 'b')")
                continue

            blobs = detector._detect_blobs(frame)
            if len(blobs) < detector.num_robots:
                print(f"Warning: detected {len(blobs)} blobs, "
                      f"expected {detector.num_robots}")

            # Assign left-to-right (by x-pixel) as tb_1, tb_2, tb_3
            blobs_sorted = sorted(blobs[:detector.num_robots],
                                  key=lambda b: b[0])
            initial_pos = {}
            for idx, (cx, cy, area) in enumerate(blobs_sorted):
                rid = idx + 1
                wx, wy = detector.pixel_to_world(cx, cy)
                initial_pos[rid] = (wx, wy)
                print(f"  tb_{rid}: pixel=({cx},{cy}) → "
                      f"world=({wx:.3f},{wy:.3f})  area={area:.0f}")

            detector.initialize_identities(blobs, initial_pos)
            identities_set = True
            print(f"Identities initialized for {len(initial_pos)} robots")

        elif key == ord("m"):
            show_mask = not show_mask
            print(f"Mask overlay: {'ON' if show_mask else 'OFF'}")

        elif key == ord("s"):
            snap_path = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(snap_path, frame)
            print(f"Saved {snap_path}")

        # Print poses (throttled)
        if identities_set and int(time.time() * 2) % 2 == 0:
            parts = []
            for rid, p in sorted(poses.items()):
                parts.append(f"tb_{rid}:({p.x:.2f},{p.y:.2f})")
            if parts:
                print("  ".join(parts), end="\r")

    cap.release()
    cv2.destroyAllWindows()
    print()


if __name__ == "__main__":
    main()
