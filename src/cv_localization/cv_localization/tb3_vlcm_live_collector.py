#!/usr/bin/env python3
"""Live TurtleBot3 WebDataset collector for MA-VLCM fine tuning."""

import argparse
import io
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
import random
import sys
import tarfile
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import rclpy
    from geometry_msgs.msg import PointStamped, PoseStamped, Twist
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.utilities import remove_ros_args
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
except Exception:  # pragma: no cover - unit tests import pure helpers
    rclpy = None
    PointStamped = None
    PoseStamped = None
    Twist = None
    Node = object
    ExternalShutdownException = Exception
    remove_ros_args = lambda args: args
    CompressedImage = None
    String = None
    Trigger = None


ROBOT_COLORS = ("red", "blue", "green")


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


@dataclass
class GoalSamplingConfig:
    workspace_width_m: float = 3.048
    workspace_height_m: float = 3.048
    wall_margin_m: float = 0.20
    goal_min_separation_m: float = 0.50
    goal_min_robot_distance_m: float = 0.35
    max_attempts: int = 500


@dataclass
class RewardConfig:
    goal_radius_m: float = 0.12
    proximity_penalty_distance_m: float = 0.20
    progress_scale: float = 1.0
    success_reward: float = 5.0
    proximity_penalty: float = -1.0
    failure_terminal_reward: float = -25.0


@dataclass
class EpisodeStep:
    key: str
    frame_png: bytes
    state: Dict
    reward: float
    episode_reward: float
    distances: np.ndarray
    adjacency: np.ndarray


@dataclass
class StuckTracker:
    start_time: Optional[float] = None
    start_distance: float = 0.0


def yaw_from_quat_xyzw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def split_csv(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def safe_bounds(config: GoalSamplingConfig) -> Tuple[float, float]:
    half_w = max(0.0, 0.5 * config.workspace_width_m - config.wall_margin_m)
    half_h = max(0.0, 0.5 * config.workspace_height_m - config.wall_margin_m)
    return half_w, half_h


def inside_safe_bounds(point: Sequence[float], config: GoalSamplingConfig) -> bool:
    half_w, half_h = safe_bounds(config)
    return -half_w <= float(point[0]) <= half_w and -half_h <= float(point[1]) <= half_h


def goals_are_valid(
    goals: Dict[str, Tuple[float, float]],
    robot_positions: Dict[str, Tuple[float, float]],
    config: GoalSamplingConfig,
) -> bool:
    names = list(goals)
    for name, goal in goals.items():
        if not inside_safe_bounds(goal, config):
            return False
        if name in robot_positions:
            if distance(goal, robot_positions[name]) < config.goal_min_robot_distance_m:
                return False
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            if distance(goals[first], goals[second]) < config.goal_min_separation_m:
                return False
    return True


def deterministic_goal_lattice(robot_names: Sequence[str], config: GoalSamplingConfig):
    half_w, half_h = safe_bounds(config)
    inset_x = min(half_w, max(config.goal_min_separation_m, half_w * 0.75))
    inset_y = min(half_h, max(config.goal_min_separation_m, half_h * 0.75))
    candidates = [
        (-inset_x, -inset_y),
        (inset_x, inset_y),
        (-inset_x, inset_y),
        (inset_x, -inset_y),
        (0.0, inset_y),
        (0.0, -inset_y),
        (-inset_x, 0.0),
        (inset_x, 0.0),
    ]
    return {
        name: candidates[i % len(candidates)]
        for i, name in enumerate(robot_names)
    }


def sample_random_goals(
    robot_names: Sequence[str],
    robot_positions: Dict[str, Tuple[float, float]],
    config: GoalSamplingConfig,
    rng: random.Random,
) -> Dict[str, Tuple[float, float]]:
    half_w, half_h = safe_bounds(config)
    for _ in range(max(1, config.max_attempts)):
        goals = {
            name: (rng.uniform(-half_w, half_w), rng.uniform(-half_h, half_h))
            for name in robot_names
        }
        if goals_are_valid(goals, robot_positions, config):
            return goals

    fallback = deterministic_goal_lattice(robot_names, config)
    if goals_are_valid(fallback, robot_positions, config):
        return fallback

    # Last resort: keep goals separated and in-bounds even if a robot starts close.
    return fallback


def distance_matrix(positions: Sequence[Tuple[float, float]]) -> np.ndarray:
    count = len(positions)
    out = np.zeros((count, count), dtype=np.float32)
    for i in range(count):
        for j in range(i + 1, count):
            d = distance(positions[i], positions[j])
            out[i, j] = d
            out[j, i] = d
    return out


def adjacency_from_distances(distances: np.ndarray, threshold_m: float = 4.0) -> np.ndarray:
    return (np.asarray(distances, dtype=np.float32) < float(threshold_m)).astype(np.float32)


def compute_step_rewards(
    previous_distances: Sequence[float],
    current_distances: Sequence[float],
    reached_before: Sequence[bool],
    reached_now: Sequence[bool],
    distances: np.ndarray,
    config: RewardConfig,
    terminal_failure: bool = False,
) -> Tuple[List[float], float]:
    rewards: List[float] = []
    for prev, current, was_reached, is_reached in zip(
        previous_distances, current_distances, reached_before, reached_now
    ):
        reward = config.progress_scale * (float(prev) - float(current))
        if is_reached and not was_reached:
            reward += config.success_reward
        rewards.append(float(reward))

    team_penalty = 0.0
    for i in range(distances.shape[0]):
        for j in range(i + 1, distances.shape[1]):
            if float(distances[i, j]) < config.proximity_penalty_distance_m:
                team_penalty += config.proximity_penalty

    scalar = float(np.mean(rewards) if rewards else 0.0) + team_penalty
    if terminal_failure:
        scalar += config.failure_terminal_reward
    return rewards, scalar


class EpisodeTarWriter:
    def __init__(self, dataset_dir: Path, episode_id: str):
        self.dataset_dir = Path(dataset_dir)
        self.episode_id = episode_id
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_path = self.dataset_dir / f"{episode_id}.tar.tmp"
        self.final_path = self.dataset_dir / f"{episode_id}.tar"
        self._tar = tarfile.open(self.tmp_path, "w")

    @staticmethod
    def _npy_bytes(array: np.ndarray) -> bytes:
        stream = io.BytesIO()
        np.save(stream, array)
        return stream.getvalue()

    def _add_bytes(self, name: str, payload: bytes) -> None:
        info = tarfile.TarInfo(name=name)
        info.size = len(payload)
        self._tar.addfile(info, io.BytesIO(payload))

    def write_step(self, step: EpisodeStep) -> None:
        prefix = step.key
        self._add_bytes(f"{prefix}.overhead.png", step.frame_png)
        self._add_bytes(
            f"{prefix}.state.json",
            json.dumps(step.state, sort_keys=True).encode("utf-8"),
        )
        self._add_bytes(f"{prefix}.reward.json", json.dumps(float(step.reward)).encode("utf-8"))
        self._add_bytes(
            f"{prefix}.episode_reward.json",
            json.dumps(float(step.episode_reward)).encode("utf-8"),
        )
        self._add_bytes(f"{prefix}.dist.npy", self._npy_bytes(step.distances))
        self._add_bytes(f"{prefix}.adj.npy", self._npy_bytes(step.adjacency))

    def close(self, keep: bool = True) -> Optional[Path]:
        if self._tar is not None:
            self._tar.close()
            self._tar = None
        if keep:
            os.replace(self.tmp_path, self.final_path)
            return self.final_path
        if self.tmp_path.exists():
            self.tmp_path.unlink()
        return None


class TB3VLCMLiveCollector(Node):
    def __init__(self):
        if rclpy is None:
            raise ImportError("rclpy and ROS 2 message packages are required.")
        super().__init__("tb3_vlcm_live_collector")
        self.robot_names = split_csv(
            self.declare_parameter("robot_names_csv", "tb_1,tb_2,tb_3").value)
        self.map_frame = str(self.declare_parameter("map_frame", "map").value)
        self.episodes_target = int(self.declare_parameter("episodes", 100).value)
        self.dataset_dir = Path(str(self.declare_parameter(
            "dataset_dir",
            "/home/adi2440/Desktop/MARL_Shahil_Aditya/MA-VLCM/data/tb3_lab",
        ).value)).expanduser()
        self.sample_rate_hz = float(self.declare_parameter("sample_rate_hz", 30.0).value)
        seed = int(self.declare_parameter("random_seed", 0).value)
        self.rng = random.Random(seed if seed != 0 else None)

        self.goal_config = GoalSamplingConfig(
            workspace_width_m=float(self.declare_parameter("workspace_width_m", 3.048).value),
            workspace_height_m=float(self.declare_parameter("workspace_height_m", 3.048).value),
            wall_margin_m=float(self.declare_parameter("wall_margin_m", 0.20).value),
            goal_min_separation_m=float(
                self.declare_parameter("goal_min_separation_m", 0.50).value),
            goal_min_robot_distance_m=float(
                self.declare_parameter("goal_min_robot_distance_m", 0.35).value),
        )
        self.reward_config = RewardConfig(
            goal_radius_m=float(self.declare_parameter("goal_radius_m", 0.12).value),
            proximity_penalty_distance_m=float(
                self.declare_parameter("proximity_penalty_distance_m", 0.20).value),
            progress_scale=float(self.declare_parameter("progress_reward_scale", 1.0).value),
            success_reward=float(self.declare_parameter("success_reward", 5.0).value),
            proximity_penalty=float(self.declare_parameter("proximity_penalty", -1.0).value),
            failure_terminal_reward=float(
                self.declare_parameter("failure_terminal_reward", -25.0).value),
        )
        self.timeout_s = float(self.declare_parameter("episode_timeout_s", 120.0).value)
        self.stuck_command_v_mps = float(self.declare_parameter("stuck_command_v_mps", 0.03).value)
        self.stuck_measured_speed_mps = float(
            self.declare_parameter("stuck_measured_speed_mps", 0.01).value)
        self.stuck_progress_m = float(self.declare_parameter("stuck_progress_m", 0.03).value)
        self.stuck_window_s = float(self.declare_parameter("stuck_window_s", 8.0).value)

        self.poses: Dict[str, Pose2D] = {name: Pose2D() for name in self.robot_names}
        self.pose_times: Dict[str, float] = {name: 0.0 for name in self.robot_names}
        self.commands: Dict[str, Tuple[float, float]] = {
            name: (0.0, 0.0) for name in self.robot_names}
        self.measured_speed: Dict[str, float] = {name: 0.0 for name in self.robot_names}
        self.measured_velocity: Dict[str, Tuple[float, float]] = {
            name: (0.0, 0.0) for name in self.robot_names}
        self.latest_frame: Optional[bytes] = None
        self.latest_frame_time = 0.0
        self.state = "waiting_for_poses"
        self.pending_goals: Dict[str, Tuple[float, float]] = {}
        self.episode_index = 0
        self.writer: Optional[EpisodeTarWriter] = None
        self.episode_id = ""
        self.episode_start = 0.0
        self.last_sample = 0.0
        self.step_index = 0
        self.previous_goal_distances: List[float] = []
        self.reached_once: List[bool] = []
        self.cumulative_reward = 0.0
        self.stuck: Dict[str, StuckTracker] = {
            name: StuckTracker() for name in self.robot_names}
        self.session_active = True
        self.last_status = ""

        self.goal_pubs = {
            name: self.create_publisher(PointStamped, f"/{name}/mppi_goal", 10)
            for name in self.robot_names
        }
        self.preview_pub = self.create_publisher(String, "/fleet_vlcm/pending_goals", 10)
        self.status_pub = self.create_publisher(String, "/fleet_vlcm/status", 10)
        self.start_client = self.create_client(Trigger, "/fleet_mppi/start")
        self.stop_client = self.create_client(Trigger, "/fleet_mppi/stop")
        self.clear_client = self.create_client(Trigger, "/fleet_mppi/clear_goals")

        for name in self.robot_names:
            self.create_subscription(
                PoseStamped,
                f"/{name}/cv_pose",
                lambda msg, robot=name: self._pose_cb(robot, msg),
                10,
            )
            self.create_subscription(
                Twist,
                f"/{name}/cmd_vel",
                lambda msg, robot=name: self._cmd_cb(robot, msg),
                10,
            )
            self.create_subscription(
                Twist,
                f"/{name}/cv_measured_velocity",
                lambda msg, robot=name: self._measured_velocity_cb(robot, msg),
                10,
            )

        self.create_subscription(CompressedImage, "/fleet_vlcm/overhead/compressed",
                                 self._frame_cb, 10)
        self.create_subscription(String, "/fleet_vlcm/key_event", self._key_cb, 10)
        self.create_subscription(String, "/fleet_mppi/status", self._fleet_status_cb, 10)
        timer_period_s = min(0.1, 0.5 / max(1.0, self.sample_rate_hz))
        self.timer = self.create_timer(timer_period_s, self._timer_cb)
        self._publish_status(
            f"VLCM collector waiting for poses; target episodes={self.episodes_target}",
            force=True,
        )

    def _pose_cb(self, robot: str, msg: PoseStamped) -> None:
        q = msg.pose.orientation
        self.poses[robot] = Pose2D(
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            yaw_from_quat_xyzw(q.x, q.y, q.z, q.w),
        )
        self.pose_times[robot] = time.monotonic()

    def _cmd_cb(self, robot: str, msg: Twist) -> None:
        self.commands[robot] = (float(msg.linear.x), float(msg.angular.z))

    def _measured_velocity_cb(self, robot: str, msg: Twist) -> None:
        self.measured_velocity[robot] = (float(msg.linear.x), float(msg.angular.z))
        self.measured_speed[robot] = float(msg.linear.z)

    def _frame_cb(self, msg: CompressedImage) -> None:
        self.latest_frame = bytes(msg.data)
        self.latest_frame_time = time.monotonic()

    def _fleet_status_cb(self, msg: String) -> None:
        text = msg.data.lower()
        if self.state == "running" and (
            "outside safe boundary" in text or "unsafe live spacing" in text
        ):
            self._finish_episode(f"controller_stop:{msg.data}", failed=True)

    def _key_cb(self, msg: String) -> None:
        key = msg.data.strip().lower()
        if key == "a" and self.state == "pending":
            self._accept_goals()
        elif key == "n" and self.state in ("pending", "waiting_for_poses"):
            self._generate_pending_goals()
        elif key == "f" and self.state == "running":
            self._finish_episode("manual_failure", failed=True)
        elif key == "x":
            self.session_active = False
            if self.state == "running":
                self._finish_episode("operator_stop", failed=True)
            self._call_trigger(self.stop_client, "stop")
            self._publish_preview(clear=True)
            self._publish_status("VLCM collection stopped by operator", force=True)

    def _timer_cb(self) -> None:
        if not self.session_active:
            return
        if self.episode_index >= self.episodes_target:
            self._publish_status("VLCM collection complete", force=True)
            self.session_active = False
            self._publish_preview(clear=True)
            return
        if self.state == "waiting_for_poses":
            if self._has_recent_poses():
                self._generate_pending_goals()
            return
        if self.state != "running":
            return

        now = time.monotonic()
        if now - self.episode_start > self.timeout_s:
            self._finish_episode("timeout", failed=True)
            return
        boundary_reason = self._boundary_failure_reason()
        if boundary_reason:
            self._finish_episode(boundary_reason, failed=True)
            return
        stuck_reason = self._stuck_failure_reason(now)
        if stuck_reason:
            self._finish_episode(stuck_reason, failed=True)
            return

        if self._all_reached():
            self._finish_episode("success", failed=False)
            return

        if now - self.last_sample >= 1.0 / max(0.1, self.sample_rate_hz):
            self._sample_step(done=False, termination_reason="")
            self.last_sample = now

    def _has_recent_poses(self) -> bool:
        now = time.monotonic()
        return all(now - self.pose_times[name] < 1.0 for name in self.robot_names)

    def _robot_positions(self) -> Dict[str, Tuple[float, float]]:
        return {name: (self.poses[name].x, self.poses[name].y) for name in self.robot_names}

    def _generate_pending_goals(self) -> None:
        self._call_trigger(self.clear_client, "clear goals")
        self.pending_goals = sample_random_goals(
            self.robot_names,
            self._robot_positions(),
            self.goal_config,
            self.rng,
        )
        self.state = "pending"
        self._publish_preview()
        self._publish_status(
            "Generated VLCM goals. Press a=accept, n=regenerate, x=stop collection.",
            force=True,
        )

    def _accept_goals(self) -> None:
        if not self.pending_goals:
            return
        for name, goal in self.pending_goals.items():
            msg = PointStamped()
            msg.header.frame_id = self.map_frame
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.point.x = float(goal[0])
            msg.point.y = float(goal[1])
            msg.point.z = 0.0
            self.goal_pubs[name].publish(msg)
        self._call_trigger(self.start_client, "start")
        self._start_episode()

    def _start_episode(self) -> None:
        self.state = "running"
        self.episode_index += 1
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.episode_id = f"tb3_lab_ep{self.episode_index:04d}_{timestamp}"
        self.writer = EpisodeTarWriter(self.dataset_dir, self.episode_id)
        self.episode_start = time.monotonic()
        self.last_sample = 0.0
        self.step_index = 0
        self.cumulative_reward = 0.0
        self.previous_goal_distances = self._goal_distances()
        self.reached_once = [d <= self.reward_config.goal_radius_m
                             for d in self.previous_goal_distances]
        self.stuck = {name: StuckTracker() for name in self.robot_names}
        self._publish_preview()
        self._publish_status(
            f"Started VLCM episode {self.episode_index}/{self.episodes_target}: "
            f"{self.episode_id}",
            force=True,
        )

    def _finish_episode(self, reason: str, failed: bool) -> None:
        if self.state != "running":
            return
        self._sample_step(done=True, termination_reason=reason, terminal_failure=failed)
        self._call_trigger(self.stop_client, "stop")
        final_path = self.writer.close(keep=True) if self.writer is not None else None
        self.writer = None
        self.state = "waiting_for_poses"
        self.pending_goals = {}
        self._publish_preview(clear=True)
        status = "failed" if failed else "completed"
        self._publish_status(
            f"VLCM episode {self.episode_index} {status}: {reason}; saved {final_path}",
            force=True,
        )

    def _sample_step(
        self,
        done: bool,
        termination_reason: str,
        terminal_failure: bool = False,
    ) -> None:
        if self.writer is None or self.latest_frame is None:
            return
        positions = [(self.poses[name].x, self.poses[name].y) for name in self.robot_names]
        dist_mat = distance_matrix(positions)
        adjacency = adjacency_from_distances(dist_mat)
        goal_distances = self._goal_distances()
        reached_now = [d <= self.reward_config.goal_radius_m for d in goal_distances]
        agent_rewards, scalar_reward = compute_step_rewards(
            self.previous_goal_distances,
            goal_distances,
            self.reached_once,
            reached_now,
            dist_mat,
            self.reward_config,
            terminal_failure=terminal_failure,
        )
        self.cumulative_reward += scalar_reward
        state = self._state_json(
            goal_distances,
            dist_mat,
            agent_rewards,
            reached_now,
            done,
            termination_reason,
            terminal_failure,
        )
        key = f"{self.episode_id}_step{self.step_index:04d}"
        self.writer.write_step(EpisodeStep(
            key=key,
            frame_png=self.latest_frame,
            state=state,
            reward=scalar_reward,
            episode_reward=self.cumulative_reward,
            distances=dist_mat,
            adjacency=adjacency,
        ))
        self.previous_goal_distances = goal_distances
        self.reached_once = [old or new for old, new in zip(self.reached_once, reached_now)]
        self.step_index += 1

    def _state_json(
        self,
        goal_distances: Sequence[float],
        dist_mat: np.ndarray,
        agent_rewards: Sequence[float],
        reached_now: Sequence[bool],
        done: bool,
        termination_reason: str,
        terminal_failure: bool,
    ) -> Dict:
        agents = []
        for i, name in enumerate(self.robot_names):
            pose = self.poses[name]
            goal = self.pending_goals.get(name, (0.0, 0.0))
            cmd = self.commands[name]
            measured = self.measured_velocity[name]
            row = dist_mat[i].copy()
            if len(row) > i:
                row[i] = np.inf
            min_neighbor = float(np.min(row)) if len(row) > 1 else 0.0
            collision = min_neighbor < self.reward_config.proximity_penalty_distance_m
            agents.append({
                "id": i,
                "domain_id": i + 1,
                "robot": name,
                "color": ROBOT_COLORS[i] if i < len(ROBOT_COLORS) else "unknown",
                "goal_label": chr(ord("A") + i),
                "goal_pos": [float(goal[0]), float(goal[1])],
                "pos": [float(pose.x), float(pose.y)],
                "yaw": float(pose.theta),
                "vel": [float(cmd[0]), float(cmd[1])],
                "measured_vel": [float(measured[0]), float(measured[1])],
                "measured_speed": float(self.measured_speed[name]),
                "dist_to_goal": float(goal_distances[i]),
                "min_neighbor_dist": min_neighbor,
                "reached": bool(reached_now[i]),
                "collision": bool(collision),
                "failure": bool(terminal_failure),
                "action": "STOP" if reached_now[i] else "FORWARD",
                "reward": float(agent_rewards[i]),
            })
        return {
            "episode_meta": {
                "episode_id": self.episode_id,
                "episode_index": self.episode_index,
                "step": self.step_index,
                "done": bool(done),
                "success": bool(done and not terminal_failure),
                "failure": bool(terminal_failure),
                "outcome": self._episode_outcome(done, terminal_failure),
                "termination_reason": termination_reason,
                "elapsed_s": float(time.monotonic() - self.episode_start),
            },
            "agents": agents,
            "reward": float(np.mean(agent_rewards) if agent_rewards else 0.0),
            "cumulative_reward": float(self.cumulative_reward),
        }

    @staticmethod
    def _episode_outcome(done: bool, terminal_failure: bool) -> str:
        if not done:
            return "running"
        return "failure" if terminal_failure else "success"

    def _goal_distances(self) -> List[float]:
        distances = []
        for name in self.robot_names:
            pose = self.poses[name]
            goal = self.pending_goals.get(name, (pose.x, pose.y))
            distances.append(distance((pose.x, pose.y), goal))
        return distances

    def _all_reached(self) -> bool:
        return all(d <= self.reward_config.goal_radius_m for d in self._goal_distances())

    def _boundary_failure_reason(self) -> str:
        for name in self.robot_names:
            pose = self.poses[name]
            if not inside_safe_bounds((pose.x, pose.y), self.goal_config):
                return f"boundary:{name}"
        return ""

    def _stuck_failure_reason(self, now: float) -> str:
        distances = self._goal_distances()
        for i, name in enumerate(self.robot_names):
            if distances[i] <= self.reward_config.goal_radius_m:
                self.stuck[name] = StuckTracker()
                continue
            commanded_v = self.commands[name][0]
            slow = self.measured_speed[name] < self.stuck_measured_speed_mps
            if commanded_v > self.stuck_command_v_mps and slow:
                tracker = self.stuck[name]
                if tracker.start_time is None:
                    tracker.start_time = now
                    tracker.start_distance = distances[i]
                progress = tracker.start_distance - distances[i]
                if progress >= self.stuck_progress_m:
                    tracker.start_time = now
                    tracker.start_distance = distances[i]
                elif now - tracker.start_time >= self.stuck_window_s:
                    return f"stuck:{name}"
            else:
                self.stuck[name] = StuckTracker()
        return ""

    def _publish_preview(self, clear: bool = False) -> None:
        msg = String()
        if clear:
            msg.data = json.dumps({"state": "clear", "goals": []})
        else:
            msg.data = json.dumps({
                "state": self.state,
                "episode_index": self.episode_index + (1 if self.state == "pending" else 0),
                "episodes_target": self.episodes_target,
                "goals": [
                    {"robot": name, "x": goal[0], "y": goal[1]}
                    for name, goal in self.pending_goals.items()
                ],
            })
        self.preview_pub.publish(msg)

    def _publish_status(self, text: str, force: bool = False) -> None:
        if not force and text == self.last_status:
            return
        self.last_status = text
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(text)

    def _call_trigger(self, client, label: str) -> None:
        if client.service_is_ready():
            client.call_async(Trigger.Request())
        else:
            self.get_logger().warning(f"Cannot {label}: service is not ready yet.")

    def stop_for_shutdown(self) -> None:
        if self.writer is not None:
            self.writer.close(keep=True)
            self.writer = None


def main(argv: Optional[Iterable[str]] = None) -> None:
    if rclpy is None:
        raise RuntimeError("ROS 2 Python packages are not available.")
    args = list(argv) if argv is not None else None
    rclpy.init(args=args)
    node = None
    try:
        node = TB3VLCMLiveCollector()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.stop_for_shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main(remove_ros_args(args=sys.argv))
