#!/usr/bin/env python3
"""
Robot detection module — marker-free version.

Detects TurtleBots as dark blobs on the checkerboard floor using:
  1. Background subtraction (reference image of empty workspace)
  2. Adaptive thresholding + contour detection
  3. Frame-to-frame nearest-neighbour identity tracking

Orientation (θ) is NOT available from vision alone — it must come
from odom fusion (odom_fusion_node.py).

Returns per-robot (x_world, y_world) using the calibrated homography.
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml


@dataclass
class RobotPose:
    """Detected pose for one robot."""
    robot_id: int
    x: float            # metres in world frame
    y: float            # metres in world frame
    theta: float         # radians — NaN when from vision only
    timestamp: float     # time.time()
    source: str          # "vision" or "aruco"
    confidence: float    # 0-1
    pixel_x: int = 0     # pixel position (for visualization)
    pixel_y: int = 0

    def as_tuple(self):
        return (self.x, self.y, self.theta)


@dataclass
class TrackedRobot:
    """Persistent tracking state for one robot."""
    robot_id: int
    smoothed_x: float = 0.0
    smoothed_y: float = 0.0
    smoothed_theta: float = 0.0
    pixel_x: int = 0
    pixel_y: int = 0
    last_seen: float = 0.0
    initialized: bool = False
    lost_count: int = 0


def _normalize_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class RobotDetector:
    """
    Detects TurtleBots in overhead camera frames without physical markers.

    Two-phase approach:
      Phase 1 (calibration): Capture background image of empty workspace.
      Phase 2 (runtime): Detect dark blobs via background subtraction +
          adaptive thresholding, track identities frame-to-frame.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.
    calibration_path : str
        Path to calibration.yaml with the homography matrix.
    """

    def __init__(self, config_path: str, calibration_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        with open(calibration_path) as f:
            cal = yaml.safe_load(f)

        self.H = np.array(cal["homography"], dtype=np.float64)
        self.H_inv = np.linalg.inv(self.H)
        self.workspace_w = cal["workspace"]["width_m"]
        self.workspace_h = cal["workspace"]["height_m"]

        # Pixel corners of workspace (for masking)
        corners = cal["pixel_corners"]
        self.workspace_pixel_corners = np.array([
            corners["top_left"],
            corners["top_right"],
            corners["bottom_right"],
            corners["bottom_left"],
        ], dtype=np.int32)

        # Detection parameters
        det_cfg = self.cfg.get("detection", {})
        self.num_robots = det_cfg.get("num_robots", 3)
        self.min_blob_area = det_cfg.get("min_blob_area", 800)
        self.max_blob_area = det_cfg.get("max_blob_area", 15000)
        self.robot_dark_thresh = det_cfg.get("robot_dark_threshold", 60)
        self.bg_diff_thresh = det_cfg.get("background_diff_threshold", 40)
        self.morph_kernel_size = det_cfg.get("morph_kernel_size", 7)

        # Tracking parameters
        track_cfg = self.cfg.get("tracking", {})
        self.alpha = track_cfg.get("smoothing_alpha", 0.7)
        self.max_stale = track_cfg.get("max_stale_sec", 2.0)
        self.max_match_dist_px = det_cfg.get("max_match_distance_px", 150)

        # Robot IDs (1-based: tb_1, tb_2, tb_3)
        self.robot_ids = list(range(1, self.num_robots + 1))

        # Tracking state
        self.tracked: Dict[int, TrackedRobot] = {
            rid: TrackedRobot(robot_id=rid) for rid in self.robot_ids
        }

        # Background image (set via set_background())
        self.bg_gray = None
        self.bg_set = False

        # Workspace mask (to ignore outside the workspace boundary)
        self._workspace_mask = None

    # ------------------------------------------------------------------
    # Background management
    # ------------------------------------------------------------------
    def set_background(self, frame: np.ndarray):
        """Store a reference image of the empty workspace."""
        self.bg_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.bg_gray = cv2.GaussianBlur(self.bg_gray, (5, 5), 0)
        self.bg_set = True

    def has_background(self) -> bool:
        return self.bg_set

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------
    def pixel_to_world(self, px: float, py: float) -> Tuple[float, float]:
        """Transform a single pixel coordinate to world (metres)."""
        pt = np.array([[[px, py]]], dtype=np.float64)
        world = cv2.perspectiveTransform(pt, self.H).flatten()
        return float(world[0]), float(world[1])

    def world_to_pixel(self, wx: float, wy: float) -> Tuple[int, int]:
        """Transform world (metres) to pixel coordinate."""
        pt = np.array([[[wx, wy]]], dtype=np.float64)
        px = cv2.perspectiveTransform(pt, self.H_inv).flatten()
        return int(px[0]), int(px[1])

    def _get_workspace_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create a mask that is white inside the workspace, black outside."""
        if self._workspace_mask is None or self._workspace_mask.shape[:2] != shape[:2]:
            self._workspace_mask = np.zeros(shape[:2], dtype=np.uint8)
            cv2.fillPoly(self._workspace_mask,
                         [self.workspace_pixel_corners], 255)
            # Erode slightly to avoid border detection
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            self._workspace_mask = cv2.erode(self._workspace_mask, kernel)
        return self._workspace_mask

    def _is_in_workspace(self, wx: float, wy: float) -> bool:
        """Check if a world coordinate is within the workspace bounds."""
        margin = 0.05  # 5cm margin
        hw = self.workspace_w / 2.0
        hh = self.workspace_h / 2.0
        return (-hw + margin < wx < hw - margin and
                -hh + margin < wy < hh - margin)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def _detect_blobs(self, frame: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        Detect dark robot-shaped blobs in the frame.

        Returns list of (cx_px, cy_px, area) for each detected blob.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        workspace_mask = self._get_workspace_mask(frame.shape)

        # --- Method 1: Background subtraction ---
        if self.bg_set:
            diff = cv2.absdiff(gray, self.bg_gray)
            _, fg_mask = cv2.threshold(diff, self.bg_diff_thresh, 255,
                                       cv2.THRESH_BINARY)
        else:
            # Fallback: just threshold for dark objects
            _, fg_mask = cv2.threshold(gray, self.robot_dark_thresh, 255,
                                       cv2.THRESH_BINARY_INV)

        # --- Method 2: Dark object detection (combined with bg sub) ---
        _, dark_mask = cv2.threshold(gray, self.robot_dark_thresh, 255,
                                     cv2.THRESH_BINARY_INV)

        # Combine: object must be dark AND different from background
        if self.bg_set:
            combined = cv2.bitwise_and(fg_mask, dark_mask)
        else:
            combined = dark_mask

        # Apply workspace mask
        combined = cv2.bitwise_and(combined, workspace_mask)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        # Store for visualization
        self._last_mask = combined

        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        blobs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_blob_area or area > self.max_blob_area:
                continue

            M = cv2.moments(cnt)
            if M["m00"] < 1:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Check if centroid is in workspace
            wx, wy = self.pixel_to_world(cx, cy)
            if not self._is_in_workspace(wx, wy):
                continue

            blobs.append((cx, cy, area))

        # Sort by area descending — take top N
        blobs.sort(key=lambda b: b[2], reverse=True)
        return blobs[:self.num_robots + 2]  # Allow a few extras for filtering

    # ------------------------------------------------------------------
    # Identity tracking (nearest-neighbour)
    # ------------------------------------------------------------------
    def initialize_identities(self, blobs: List[Tuple[int, int, float]],
                               initial_positions: Dict[int, Tuple[float, float]]):
        """
        One-time assignment: match detected blobs to known starting
        positions from the mission CSV.

        Parameters
        ----------
        blobs : detected blob centroids [(cx, cy, area), ...]
        initial_positions : {robot_id: (x_world, y_world)}
        """
        # Convert blob pixels to world coords
        blob_worlds = []
        for cx, cy, area in blobs:
            wx, wy = self.pixel_to_world(cx, cy)
            blob_worlds.append((wx, wy, cx, cy))

        # Greedy nearest-neighbour assignment
        used_blobs = set()
        for rid, (target_x, target_y) in initial_positions.items():
            best_idx = -1
            best_dist = float("inf")
            for i, (wx, wy, cx, cy) in enumerate(blob_worlds):
                if i in used_blobs:
                    continue
                dist = math.hypot(wx - target_x, wy - target_y)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx >= 0:
                wx, wy, cx, cy = blob_worlds[best_idx]
                used_blobs.add(best_idx)
                tr = self.tracked[rid]
                tr.smoothed_x = wx
                tr.smoothed_y = wy
                tr.pixel_x = cx
                tr.pixel_y = cy
                tr.last_seen = time.time()
                tr.initialized = True

    def _match_blobs_to_tracks(self, blobs: List[Tuple[int, int, float]]):
        """
        Frame-to-frame nearest-neighbour matching of blobs to existing tracks.
        Uses pixel distance for speed.
        """
        now = time.time()
        active_tracks = {rid: tr for rid, tr in self.tracked.items()
                         if tr.initialized}

        if not active_tracks:
            return {}

        # Build cost matrix: (track_id, blob_idx) → pixel distance
        assignments = {}
        used_blobs = set()

        # Sort tracks by how recently they were seen (prioritize fresh)
        sorted_tracks = sorted(active_tracks.items(),
                               key=lambda kv: kv[1].last_seen, reverse=True)

        for rid, tr in sorted_tracks:
            best_idx = -1
            best_dist = float("inf")

            for i, (cx, cy, area) in enumerate(blobs):
                if i in used_blobs:
                    continue
                dist = math.hypot(cx - tr.pixel_x, cy - tr.pixel_y)
                if dist < best_dist and dist < self.max_match_dist_px:
                    best_dist = dist
                    best_idx = i

            if best_idx >= 0:
                used_blobs.add(best_idx)
                assignments[rid] = blobs[best_idx]

        return assignments

    # ------------------------------------------------------------------
    # Main detect + track
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> Dict[int, RobotPose]:
        """
        Run full detection pipeline.  Returns smoothed poses for all
        detected robots as {robot_id: RobotPose}.
        """
        blobs = self._detect_blobs(frame)

        # Match to existing tracks
        assignments = self._match_blobs_to_tracks(blobs)

        now = time.time()
        output = {}

        for rid, (cx, cy, area) in assignments.items():
            wx, wy = self.pixel_to_world(cx, cy)
            tr = self.tracked[rid]

            # EMA smoothing
            a = self.alpha
            tr.smoothed_x = a * wx + (1 - a) * tr.smoothed_x
            tr.smoothed_y = a * wy + (1 - a) * tr.smoothed_y
            tr.pixel_x = cx
            tr.pixel_y = cy
            tr.last_seen = now
            tr.lost_count = 0

            output[rid] = RobotPose(
                robot_id=rid,
                x=tr.smoothed_x,
                y=tr.smoothed_y,
                theta=tr.smoothed_theta,  # From odom fusion
                timestamp=now,
                source="vision",
                confidence=0.9,
                pixel_x=cx,
                pixel_y=cy,
            )

        # Include recently-lost tracks (keeps publishing stale poses briefly)
        for rid, tr in self.tracked.items():
            if rid not in output and tr.initialized:
                age = now - tr.last_seen
                if age < self.max_stale:
                    tr.lost_count += 1
                    output[rid] = RobotPose(
                        robot_id=rid,
                        x=tr.smoothed_x,
                        y=tr.smoothed_y,
                        theta=tr.smoothed_theta,
                        timestamp=tr.last_seen,
                        source="vision(stale)",
                        confidence=max(0.1, 0.9 - age * 0.4),
                        pixel_x=tr.pixel_x,
                        pixel_y=tr.pixel_y,
                    )

        return output

    def update_orientation(self, robot_id: int, theta: float):
        """
        Called externally (by odom fusion) to update a robot's heading.
        The vision system only provides (x, y).
        """
        if robot_id in self.tracked:
            self.tracked[robot_id].smoothed_theta = theta

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def draw_detections(self, frame: np.ndarray,
                        poses: Dict[int, RobotPose],
                        show_mask: bool = False) -> np.ndarray:
        """Draw detection overlays on a copy of the frame."""
        vis = frame.copy()

        # Optionally show the detection mask in corner
        if show_mask and hasattr(self, "_last_mask"):
            mask_small = cv2.resize(self._last_mask, (320, 240))
            mask_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            vis[10:250, 10:330] = mask_bgr
            cv2.rectangle(vis, (10, 10), (330, 250), (0, 255, 255), 2)
            cv2.putText(vis, "Detection Mask", (15, 268),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Robot colors
        COLORS = {
            1: (0, 255, 0),     # green
            2: (255, 180, 0),   # cyan-ish
            3: (0, 180, 255),   # orange
        }

        for rid, pose in sorted(poses.items()):
            color = COLORS.get(rid, (255, 255, 255))
            px, py = pose.pixel_x, pose.pixel_y

            # Circle around robot
            cv2.circle(vis, (px, py), 20, color, 3)

            # Label
            label = f"tb_{rid} ({pose.x:.2f},{pose.y:.2f})"
            cv2.putText(vis, label, (px + 25, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Confidence / source
            info = f"[{pose.source}] conf={pose.confidence:.1f}"
            cv2.putText(vis, info, (px + 25, py + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Heading arrow (only if θ has been set from odom)
            if not math.isnan(pose.theta) and pose.theta != 0.0:
                arrow_len = 0.25  # metres
                end_x = pose.x + arrow_len * math.cos(pose.theta)
                end_y = pose.y + arrow_len * math.sin(pose.theta)
                epx, epy = self.world_to_pixel(end_x, end_y)
                cv2.arrowedLine(vis, (px, py), (epx, epy), color, 2,
                                tipLength=0.3)
                deg = math.degrees(pose.theta)
                cv2.putText(vis, f"{deg:.0f} deg", (px + 25, py + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return vis
