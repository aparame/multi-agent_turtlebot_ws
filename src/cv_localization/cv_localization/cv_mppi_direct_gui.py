#!/usr/bin/env python3
"""Camera GUI for marker-free CV localization and direct MPPI control."""

import argparse
import math
import os
import sys
import time

import cv2
import numpy as np
import yaml

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PointStamped, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

from cv_localization.detector import ROBOT_COLORS, RobotDetector


COLORS = ROBOT_COLORS


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class PoseEKF:
    """Small EKF: camera updates x/y, odom yaw deltas update theta."""

    def __init__(self, x, y, theta, cfg):
        self.state = np.array([x, y, normalize_angle(theta)], dtype=np.float64)
        initial_xy_std = float(cfg.get('initial_xy_std_m', 0.05))
        initial_yaw_std = float(cfg.get('initial_yaw_std_rad', 0.25))
        self.position_process_var = float(cfg.get('position_process_std_m', 0.02)) ** 2
        self.yaw_process_var = float(cfg.get('yaw_process_std_rad', 0.03)) ** 2
        self.position_measurement_var = float(cfg.get('position_measurement_std_m', 0.035)) ** 2
        self.cov = np.diag([
            initial_xy_std ** 2,
            initial_xy_std ** 2,
            initial_yaw_std ** 2,
        ]).astype(np.float64)

    def predict_yaw_delta(self, delta_yaw):
        self.state[2] = normalize_angle(self.state[2] + delta_yaw)
        self.cov += np.diag([
            self.position_process_var,
            self.position_process_var,
            self.yaw_process_var,
        ])

    def update_position(self, x_meas, y_meas):
        h = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        z = np.array([x_meas, y_meas], dtype=np.float64)
        r = np.eye(2, dtype=np.float64) * self.position_measurement_var
        innovation = z - h @ self.state
        s = h @ self.cov @ h.T + r
        k = self.cov @ h.T @ np.linalg.inv(s)
        self.state = self.state + k @ innovation
        self.state[2] = normalize_angle(self.state[2])
        self.cov = (np.eye(3, dtype=np.float64) - k @ h) @ self.cov

    @property
    def x(self):
        return float(self.state[0])

    @property
    def y(self):
        return float(self.state[1])

    @property
    def theta(self):
        return float(self.state[2])


class CVMppiDirectGui(Node):
    def __init__(self, config_path, calibration_path, background_path):
        super().__init__('cv_mppi_direct_gui')
        self.config_path = config_path
        self.calibration_path = calibration_path
        self.background_path = background_path

        if not os.path.exists(background_path):
            raise RuntimeError(
                f'Fixed background image is missing: {background_path}. '
                'Capture it once during setup before running direct control.')

        self.detector = RobotDetector(config_path, calibration_path)
        background = cv2.imread(background_path)
        if background is None:
            raise RuntimeError(f'Could not load fixed background image: {background_path}')
        self.detector.set_background(background)

        with open(config_path) as stream:
            cfg = yaml.safe_load(stream)
        cam_cfg = cfg.get('camera', {})
        ros_cfg = cfg.get('ros', {})
        track_cfg = cfg.get('tracking', {})
        workspace_cfg = cfg.get('workspace', {})
        mppi_cfg = cfg.get('mppi', {})
        self.fusion_cfg = cfg.get('fusion', {})

        self.global_frame = ros_cfg.get('global_frame', 'map')
        self.base_pattern = ros_cfg.get('base_frame_pattern', 'tb_{id}/base_footprint')
        pose_pattern = ros_cfg.get('pose_topic_pattern', '/tb_{id}/cv_pose')
        self.robot_ids = list(self.detector.robot_ids)
        self.boundary_margin_m = float(
            mppi_cfg.get('wall_margin_m', ros_cfg.get('boundary_margin_m', 0.20)))
        self.workspace_width_m = float(workspace_cfg.get('width_m', self.detector.workspace_w))
        self.workspace_height_m = float(workspace_cfg.get('height_m', self.detector.workspace_h))

        self.cap = cv2.VideoCapture(cam_cfg.get('device', 0))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {cam_cfg.get('device', 0)}")

        fourcc = cam_cfg.get('fourcc')
        if fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get('width', 1280))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get('height', 960))
        self.cap.set(cv2.CAP_PROP_FPS, cam_cfg.get('fps', 30))
        for _ in range(5):
            self.cap.read()

        self.tf_broadcaster = TransformBroadcaster(self)
        self.pose_pubs = {
            rid: self.create_publisher(
                PoseStamped,
                pose_pattern.replace('{id}', str(rid)),
                10,
            )
            for rid in self.robot_ids
        }
        self.goal_pubs = {
            rid: self.create_publisher(PointStamped, f'/tb_{rid}/mppi_goal', 10)
            for rid in self.robot_ids
        }

        self.odom_yaw = {}
        self.prev_odom_yaw = {}
        for rid in self.robot_ids:
            self.create_subscription(
                Odometry,
                f'/tb_{rid}/odom',
                lambda msg, r=rid: self._odom_cb(r, msg),
                10,
            )

        self.paths = {rid: [] for rid in self.robot_ids}
        self.offline_paths = {rid: [] for rid in self.robot_ids}
        for rid in self.robot_ids:
            self.create_subscription(
                Path,
                f'/tb_{rid}/mppi_plan',
                lambda msg, r=rid: self._path_cb(r, msg),
                10,
            )
            self.create_subscription(
                Path,
                f'/tb_{rid}/offline_plan',
                lambda msg, r=rid: self._offline_path_cb(r, msg),
                10,
            )

        self.start_client = self.create_client(Trigger, '/fleet_mppi/start')
        self.stop_client = self.create_client(Trigger, '/fleet_mppi/stop')
        self.clear_client = self.create_client(Trigger, '/fleet_mppi/clear_goals')
        self.plan_client = self.create_client(Trigger, '/fleet_mppi/plan')

        self.identities_initialized = False
        self.next_identity_index = 0
        self.assigned_identity_pixels = []
        self.assigned_headings = {}
        self.pending_heading = None
        self.ekfs = {}
        self.goals = {}
        self.latest_blobs = []
        self.latest_poses = {}
        self.last_goal_publish = 0.0
        self.should_exit = False

        self.window_name = 'CV Scheduled Multi-Agent Control'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.setMouseCallback(self.window_name, self._mouse_cb)

        rate = float(track_cfg.get('publish_rate_hz', 30))
        self.timer = self.create_timer(1.0 / rate, self._on_timer)
        self.get_logger().info(
            'CV direct GUI ready. Fixed background loaded once from '
            f'{background_path}')

    def _odom_cb(self, robot_id, msg):
        odom_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        if robot_id in self.prev_odom_yaw:
            delta = normalize_angle(odom_yaw - self.prev_odom_yaw[robot_id])
            ekf = self.ekfs.get(robot_id)
            if ekf is not None:
                ekf.predict_yaw_delta(delta)
                self.detector.update_orientation(robot_id, ekf.theta)
        self.prev_odom_yaw[robot_id] = odom_yaw
        self.odom_yaw[robot_id] = odom_yaw

    def _path_cb(self, robot_id, msg):
        self.paths[robot_id] = [
            (pose.pose.position.x, pose.pose.position.y)
            for pose in msg.poses
        ]

    def _offline_path_cb(self, robot_id, msg):
        self.offline_paths[robot_id] = [
            (pose.pose.position.x, pose.pose.position.y)
            for pose in msg.poses
        ]

    def _mouse_cb(self, event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if not self.identities_initialized:
            if self.pending_heading is not None:
                self._handle_heading_click(x, y)
                return
            self._handle_identity_click(x, y)
            return

        if len(self.goals) >= len(self.robot_ids):
            self.get_logger().info('All goals are already set. Press r to clear them.')
            return

        wx, wy = self.detector.pixel_to_world(x, y)
        if not self._is_inside_safe_boundary(wx, wy):
            self.get_logger().warning(
                f'Ignored goal outside safe boundary: ({wx:.2f}, {wy:.2f}); '
                f'keep goals at least {self.boundary_margin_m:.2f} m from edges')
            return

        rid = self.robot_ids[len(self.goals)]
        self.goals[rid] = (wx, wy)
        self._publish_goal(rid)
        self.get_logger().info(f'Set tb_{rid} goal: x={wx:.3f}, y={wy:.3f}')

        if len(self.goals) == len(self.robot_ids):
            self._call_trigger(self.plan_client, 'plan')

    def _handle_identity_click(self, x, y):
        if self.pending_heading is not None or self.next_identity_index >= len(self.robot_ids):
            return
        if not self.latest_blobs:
            self.get_logger().warning('No robot blobs detected yet; cannot assign identity.')
            return

        best_blob = None
        best_dist = float('inf')
        for cx, cy, area in self.latest_blobs:
            dist = math.hypot(cx - x, cy - y)
            if dist < best_dist:
                best_blob = (cx, cy, area)
                best_dist = dist

        if best_blob is None or best_dist > 90.0:
            self.get_logger().warning('Click closer to a detected robot blob.')
            return

        for px, py in self.assigned_identity_pixels:
            if math.hypot(best_blob[0] - px, best_blob[1] - py) < 35.0:
                self.get_logger().warning('That blob is already assigned. Click a different robot.')
                return

        rid = self.robot_ids[self.next_identity_index]
        cx, cy, _area = best_blob
        wx, wy = self.detector.pixel_to_world(cx, cy)
        self.detector.initialize_track(rid, wx, wy, cx, cy, time.time())

        self.assigned_identity_pixels.append((cx, cy))
        self.pending_heading = {
            'robot_id': rid,
            'world_x': wx,
            'world_y': wy,
            'pixel_x': cx,
            'pixel_y': cy,
        }
        self.get_logger().info(
            f'Assigned clicked blob to tb_{rid}: x={wx:.3f}, y={wy:.3f}. '
            'Now click a point in the direction this robot is facing.')

    def _handle_heading_click(self, x, y):
        pending = self.pending_heading
        if pending is None:
            return
        rid = pending['robot_id']
        wx, wy = self.detector.pixel_to_world(x, y)
        dx = wx - pending['world_x']
        dy = wy - pending['world_y']
        if math.hypot(dx, dy) < 0.05:
            self.get_logger().warning('Heading click is too close to robot center.')
            return

        theta = math.atan2(dy, dx)
        self.detector.update_orientation(rid, theta)
        self.ekfs[rid] = PoseEKF(pending['world_x'], pending['world_y'], theta, self.fusion_cfg)
        if rid in self.odom_yaw:
            self.prev_odom_yaw[rid] = self.odom_yaw[rid]
        self.assigned_headings[rid] = (
            pending['pixel_x'],
            pending['pixel_y'],
            x,
            y,
        )
        self.pending_heading = None
        self.next_identity_index += 1
        self.get_logger().info(f'Initial tb_{rid} yaw estimate set to {math.degrees(theta):.1f} deg')

        if self.next_identity_index == len(self.robot_ids):
            self.identities_initialized = True
            self.get_logger().info('Robot identities locked. Click goals in order tb_1, tb_2, tb_3.')

    def _on_timer(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning('Camera read failed')
            return

        if not self.identities_initialized:
            self.latest_blobs = self.detector._detect_blobs(frame)
            vis = self._draw_identity_view(frame)
        else:
            detected_poses = self.detector.detect(frame)
            self.latest_poses = {}
            stamp = self.get_clock().now().to_msg()
            for rid, pose in detected_poses.items():
                ekf = self.ekfs.get(rid)
                if ekf is not None:
                    ekf.update_position(pose.x, pose.y)
                    pose.x = ekf.x
                    pose.y = ekf.y
                    pose.theta = ekf.theta
                    self.detector.update_orientation(rid, ekf.theta)
                self.latest_poses[rid] = pose
                self._publish_tf(rid, pose, stamp)
                self._publish_pose(rid, pose, stamp)
            self._republish_goals()
            vis = self.detector.draw_detections(frame, self.latest_poses, show_mask=False)
            self._draw_goals_and_paths(vis)
            self._draw_status(vis)

        cv2.imshow(self.window_name, vis)
        key = cv2.waitKey(1) & 0xFF
        self._handle_key(key)

        if self.should_exit:
            raise KeyboardInterrupt

    def _draw_identity_view(self, frame):
        vis = frame.copy()
        for cx, cy, area in self.latest_blobs:
            cv2.circle(vis, (cx, cy), 22, (255, 255, 255), 2)
            cv2.putText(
                vis,
                f'{int(area)} px',
                (cx + 24, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
        for idx, (px, py) in enumerate(self.assigned_identity_pixels):
            rid = self.robot_ids[idx]
            color = COLORS.get(rid, (255, 255, 255))
            cv2.circle(vis, (px, py), 28, color, 3)
            cv2.putText(vis, f'tb_{rid}', (px + 30, py - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if rid in self.assigned_headings:
                sx, sy, ex, ey = self.assigned_headings[rid]
                cv2.arrowedLine(vis, (sx, sy), (ex, ey), color, 3, tipLength=0.25)
        if self.pending_heading is not None:
            sx = self.pending_heading['pixel_x']
            sy = self.pending_heading['pixel_y']
            cv2.circle(vis, (sx, sy), 34, (255, 255, 255), 2)
        self._draw_status(vis)
        return vis

    def _draw_status(self, vis):
        lines = []
        if self.pending_heading is not None:
            lines.append(f"Click heading direction for tb_{self.pending_heading['robot_id']}")
        elif not self.identities_initialized:
            rid = self.robot_ids[self.next_identity_index]
            lines.append(f'Click robot identity: tb_{rid}')
        elif len(self.goals) < len(self.robot_ids):
            rid = self.robot_ids[len(self.goals)]
            lines.append(
                f'Click goal for tb_{rid} (>= {self.boundary_margin_m:.2f} m from boundary)')
        else:
            lines.append('Goals ready. Enter=start, Space/Esc=stop, r=clear goals, q=quit')
        lines.append('Fusion: camera x/y + clicked initial yaw + odom yaw deltas')
        lines.append('Control: direct velocity + runtime safety filter')
        lines.append('Fixed background mode: runtime recapture disabled')

        y = 30
        for line in lines:
            cv2.putText(vis, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 0), 4)
            cv2.putText(vis, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 255, 255), 2)
            y += 30

    def _draw_goals_and_paths(self, vis):
        self._draw_safe_boundary(vis)
        for rid, points in self.offline_paths.items():
            self._draw_polyline(vis, rid, points, thickness=2)

        for rid, points in self.paths.items():
            self._draw_polyline(vis, rid, points, thickness=4)

        for rid, (wx, wy) in self.goals.items():
            color = COLORS.get(rid, (255, 255, 255))
            px, py = self.detector.world_to_pixel(wx, wy)
            cv2.drawMarker(vis, (px, py), color, markerType=cv2.MARKER_CROSS,
                           markerSize=28, thickness=3)
            cv2.circle(vis, (px, py), 18, color, 2)
            cv2.putText(vis, f'tb_{rid} goal', (px + 22, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        for rid, pose in self.latest_poses.items():
            color = COLORS.get(rid, (255, 255, 255))
            start = self.detector.world_to_pixel(pose.x, pose.y)
            end = self.detector.world_to_pixel(
                pose.x + 0.25 * math.cos(pose.theta),
                pose.y + 0.25 * math.sin(pose.theta),
            )
            cv2.arrowedLine(vis, start, end, color, 3, tipLength=0.25)

    def _draw_polyline(self, vis, rid, points, thickness):
        if len(points) < 2:
            return
        color = COLORS.get(rid, (255, 255, 255))
        pixel_points = []
        for wx, wy in points:
            try:
                pixel_points.append(self.detector.world_to_pixel(wx, wy))
            except Exception:
                pass
        if len(pixel_points) >= 2:
            pts = np.array(pixel_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], False, color, thickness)

    def _draw_safe_boundary(self, vis):
        half_w = self.workspace_width_m / 2.0 - self.boundary_margin_m
        half_h = self.workspace_height_m / 2.0 - self.boundary_margin_m
        if half_w <= 0.0 or half_h <= 0.0:
            return
        corners = [
            self.detector.world_to_pixel(-half_w, half_h),
            self.detector.world_to_pixel(half_w, half_h),
            self.detector.world_to_pixel(half_w, -half_h),
            self.detector.world_to_pixel(-half_w, -half_h),
        ]
        pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], True, (255, 255, 255), 2)

    def _is_inside_safe_boundary(self, wx, wy):
        half_w = self.workspace_width_m / 2.0 - self.boundary_margin_m
        half_h = self.workspace_height_m / 2.0 - self.boundary_margin_m
        return -half_w <= wx <= half_w and -half_h <= wy <= half_h

    def _handle_key(self, key):
        if key in (255, -1):
            return
        if key in (13, 10):
            if len(self.goals) == len(self.robot_ids):
                self._call_trigger(self.start_client, 'start')
            else:
                self.get_logger().warning('Set all three goals before starting.')
        elif key in (27, 32):
            self._call_trigger(self.stop_client, 'stop')
        elif key == ord('r'):
            self.goals.clear()
            self.paths = {rid: [] for rid in self.robot_ids}
            self.offline_paths = {rid: [] for rid in self.robot_ids}
            self._call_trigger(self.clear_client, 'clear goals')
            self.get_logger().info('Cleared goals. Fixed background remains unchanged.')
        elif key == ord('q'):
            self._call_trigger(self.stop_client, 'stop')
            self.should_exit = True

    def _republish_goals(self):
        now = time.monotonic()
        if now - self.last_goal_publish < 1.0:
            return
        for rid in self.goals:
            self._publish_goal(rid)
        self.last_goal_publish = now

    def _publish_goal(self, rid):
        wx, wy = self.goals[rid]
        msg = PointStamped()
        msg.header.frame_id = self.global_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x = wx
        msg.point.y = wy
        msg.point.z = 0.0
        self.goal_pubs[rid].publish(msg)

    def _publish_tf(self, robot_id, pose, stamp):
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.global_frame
        transform.child_frame_id = self.base_pattern.replace('{id}', str(robot_id))
        transform.transform.translation.x = pose.x
        transform.transform.translation.y = pose.y
        transform.transform.translation.z = 0.0
        qx, qy, qz, qw = yaw_to_quaternion(pose.theta)
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(transform)

    def _publish_pose(self, robot_id, pose, stamp):
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.global_frame
        msg.pose.position.x = pose.x
        msg.pose.position.y = pose.y
        msg.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quaternion(pose.theta)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.pose_pubs[robot_id].publish(msg)

    def _call_trigger(self, client, label):
        if not client.service_is_ready():
            self.get_logger().warning(f'Cannot {label}: service is not ready yet.')
            return
        client.call_async(Trigger.Request())

    def destroy_node(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def default_config_paths():
    share = get_package_share_directory('cv_localization')
    config_dir = os.path.join(share, 'config')
    return (
        os.path.join(config_dir, 'config.yaml'),
        os.path.join(config_dir, 'calibration.yaml'),
        os.path.join(config_dir, 'background.jpg'),
    )


def main(args=None):
    default_config, default_calibration, default_background = default_config_paths()
    parser = argparse.ArgumentParser(description='CV localization GUI for direct MPPI control.')
    parser.add_argument('--config', default=default_config)
    parser.add_argument('--calibration', default=default_calibration)
    parser.add_argument('--background', default=default_background)
    parsed = parser.parse_args(remove_ros_args(args=args)[1:])

    for path in (parsed.config, parsed.calibration):
        if not os.path.exists(path):
            print(f'Error: required file not found: {path}', file=sys.stderr)
            sys.exit(1)

    rclpy.init(args=args)
    node = CVMppiDirectGui(parsed.config, parsed.calibration, parsed.background)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
