#!/usr/bin/env python3
"""ROS 2 RL checkpoint controller for the CV TurtleBot go-to-goal workflow."""

import math
from pathlib import Path
import sys
import time
from dataclasses import dataclass, field
import types
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import rclpy
    from geometry_msgs.msg import PointStamped, PoseStamped, Twist
    from nav_msgs.msg import Odometry, Path as RosPath
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.utilities import remove_ros_args
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
except Exception:  # pragma: no cover - unit tests can import pure helpers
    rclpy = None
    PointStamped = None
    PoseStamped = None
    Twist = None
    Odometry = None
    RosPath = None
    ExternalShutdownException = Exception
    Node = object
    String = None
    Trigger = None
    remove_ros_args = lambda args: args


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat_xyzw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def split_csv(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


@dataclass
class RlControllerParams:
    robot_names: List[str] = field(default_factory=lambda: ["tb_1", "tb_2", "tb_3"])
    map_frame: str = "map"
    control_rate_hz: float = 20.0
    pose_timeout_s: float = 0.5
    min_live_spacing_m: float = 0.25
    room_size_m: float = 3.048
    goal_radius_m: float = 0.12
    goal_termination_v_mps: float = 0.0
    goal_termination_w_radps: float = 0.0
    max_v_mps: float = 0.1
    max_w_radps: float = 1.0
    max_dv_step: float = 0.03
    max_dw_step: float = 0.18
    safety_distance_m: float = 0.25
    wall_margin_m: float = 0.20
    boundary_slowdown_margin_m: float = 0.15
    orca_filter_enabled: bool = True
    orca_radius_m: float = 0.20
    orca_neighbor_dist_m: float = 1.5
    orca_time_horizon_s: float = 3.0
    orca_avoidance_gain: float = 3.0
    orca_priority_enabled: bool = True
    orca_priority_strength: float = 5.0
    orca_priority_fixed_bias: float = 0.25
    orca_priority_min_share: float = 0.10
    sight_range: float = 9.0
    model_dir: str = ""
    aero_marl_root: str = "/home/adi2440/Desktop/MARL_Shahil_Aditya/AERO-MARL"
    policy_device: str = "auto"
    algorithm_name: str = "mappo_dgnn_dsgd"
    experiment_name: str = "single"
    seed: int = 0
    iterations: int = 4
    n_embd: int = 128
    num_heads: int = 1
    num_layers: int = 2
    n_quants: int = 1
    truely_distributed: bool = True
    truely_distributed_gnn: bool = True
    consensus_loss: bool = True
    gnn_loss_coef: float = 10.0
    critic_lr: float = 2.5e-4
    lr: float = 2.5e-4
    value_loss_coef: float = 1.0
    max_grad_norm: float = 0.6
    gamma: float = 0.95
    gae_lambda: float = 0.8
    entropy_coef: float = 0.001
    clip_param: float = 0.2
    allow_partial_restore: bool = True
    clone_extra_agents_from: int = 0
    use_comm_action: bool = True
    comm_action_dim: int = 3


@dataclass
class RobotRuntime:
    pose: Pose2D = field(default_factory=Pose2D)
    previous_pose: Optional[Pose2D] = None
    goal: Optional[Pose2D] = None
    pose_time: float = 0.0
    previous_pose_time: float = 0.0
    has_pose: bool = False
    has_goal: bool = False
    last_cmd: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    measured_cv_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    measured_cv_lateral_mps: float = 0.0
    measured_cv_speed_mps: float = 0.0
    measured_odom_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    has_odom_velocity: bool = False
    priority_wait_s: float = 0.0
    cmd_pub: object = None
    measured_velocity_pub: object = None
    path_pub: object = None


def obs_dim_for_agents(num_agents: int, include_neighbor_obs: bool = True) -> int:
    if include_neighbor_obs:
        return 5 + max(0, num_agents - 1) * 6
    return 5 + max(0, num_agents - 1)


def state_dim_for_agents(num_agents: int) -> int:
    return num_agents * 8


def poses_to_array(poses: Sequence[Pose2D]) -> np.ndarray:
    return np.asarray([[p.x, p.y, p.theta] for p in poses], dtype=np.float32)


def goals_to_array(goals: Sequence[Pose2D]) -> np.ndarray:
    return np.asarray([[g.x, g.y, g.theta] for g in goals], dtype=np.float32)


def build_policy_inputs(
    poses: Sequence[Pose2D],
    goals: Sequence[Pose2D],
    last_cmds: np.ndarray,
    params: RlControllerParams,
    comm_modes: Optional[np.ndarray] = None,
    dead: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose_arr = poses_to_array(poses)
    goal_arr = goals_to_array(goals)
    obs = build_observations(
        pose_arr,
        goal_arr,
        last_cmds,
        visibility_range=params.sight_range,
        dead=dead,
    )
    state = build_shared_state(pose_arr, goal_arr, last_cmds)
    share_obs = np.repeat(state[None, :], len(poses), axis=0).astype(np.float32)
    edge_index = build_edge_index_matrix(
        pose_arr,
        visibility_range=params.sight_range,
        comm_modes=comm_modes,
        dead=dead,
    )
    return obs, share_obs, edge_index


def build_observations(
    poses: np.ndarray,
    goals: np.ndarray,
    last_cmds: np.ndarray,
    visibility_range: float,
    dead: Optional[np.ndarray] = None,
) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float32)
    goals = np.asarray(goals, dtype=np.float32)
    last_cmds = np.asarray(last_cmds, dtype=np.float32)
    num_agents = poses.shape[0]
    if dead is None:
        dead = np.zeros(num_agents, dtype=bool)

    obs = []
    headings = poses[:, 2]
    for i in range(num_agents):
        pos = poses[i]
        rel_goal = goals[i, :2] - pos[:2]
        goal_dist = float(np.linalg.norm(rel_goal))
        bearing = math.atan2(float(rel_goal[1]), float(rel_goal[0]))
        bearing_error = wrap_to_pi(bearing - float(pos[2]))
        v_cmd, w_cmd = last_cmds[i]
        entry = [
            goal_dist,
            math.sin(bearing_error),
            math.cos(bearing_error),
            float(v_cmd),
            float(w_cmd),
        ]

        ego_heading = float(headings[i])
        ego_forward = np.array([math.cos(ego_heading), math.sin(ego_heading)], dtype=np.float32)
        ego_vel = float(v_cmd) * ego_forward
        neighbor_ids = [j for j in range(num_agents) if j != i]
        neighbor_ids.sort(key=lambda j: np.linalg.norm(poses[j, :2] - poses[i, :2]))
        for j in neighbor_ids:
            rel_pos = poses[j, :2] - poses[i, :2]
            dist = float(np.linalg.norm(rel_pos))
            if dist < visibility_range:
                bearing = math.atan2(float(rel_pos[1]), float(rel_pos[0]))
                bearing_error = wrap_to_pi(bearing - float(pos[2]))
                neighbor_alive = 0.0 if dead[j] else 1.0
                other_v_cmd = float(last_cmds[j, 0])
                if neighbor_alive > 0.0:
                    other_heading = float(headings[j])
                    other_vel = other_v_cmd * np.array(
                        [math.cos(other_heading), math.sin(other_heading)],
                        dtype=np.float32,
                    )
                    other_forward = float(np.dot(other_vel, ego_forward))
                else:
                    other_vel = np.zeros(2, dtype=np.float32)
                    other_forward = 0.0
                if dist > 1e-8:
                    los_unit = rel_pos / dist
                    closing_speed = float(-np.dot(other_vel - ego_vel, los_unit))
                else:
                    closing_speed = 0.0
                entry.extend(
                    [
                        dist,
                        math.sin(bearing_error),
                        math.cos(bearing_error),
                        other_forward,
                        closing_speed,
                        neighbor_alive,
                    ]
                )
            else:
                entry.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        obs.append(entry)
    return np.asarray(obs, dtype=np.float32)


def build_shared_state(
    poses: np.ndarray,
    goals: np.ndarray,
    last_cmds: np.ndarray,
) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float32)
    goals = np.asarray(goals, dtype=np.float32)
    last_cmds = np.asarray(last_cmds, dtype=np.float32)
    values = []
    for i in range(poses.shape[0]):
        rel = goals[i, :2] - poses[i, :2]
        goal_dist = float(np.linalg.norm(rel))
        bearing = math.atan2(float(rel[1]), float(rel[0]))
        bearing_error = wrap_to_pi(bearing - float(poses[i, 2]))
        values.extend(
            [
                float(poses[i, 0]),
                float(poses[i, 1]),
                float(poses[i, 2]),
                goal_dist,
                math.sin(bearing_error),
                math.cos(bearing_error),
                float(last_cmds[i, 0]),
                float(last_cmds[i, 1]),
            ]
        )
    return np.asarray(values, dtype=np.float32)


def build_visibility_matrix(
    poses: np.ndarray,
    visibility_range: float,
    dead: Optional[np.ndarray] = None,
) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float32)
    num_agents = poses.shape[0]
    if dead is None:
        dead = np.zeros(num_agents, dtype=bool)
    visibility = np.zeros((num_agents, num_agents), dtype=np.bool_)
    for i in range(num_agents):
        if dead[i]:
            continue
        for j in range(i + 1, num_agents):
            if dead[j]:
                continue
            dist = np.linalg.norm(poses[i, :2] - poses[j, :2])
            if dist < visibility_range:
                visibility[i, j] = True
                visibility[j, i] = True
    return visibility


def build_edge_index_matrix(
    poses: np.ndarray,
    visibility_range: float,
    comm_modes: Optional[np.ndarray] = None,
    dead: Optional[np.ndarray] = None,
) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float32)
    num_agents = poses.shape[0]
    if dead is None:
        dead = np.zeros(num_agents, dtype=bool)
    visibility = build_visibility_matrix(poses, visibility_range, dead)
    if comm_modes is not None:
        modes = np.asarray(comm_modes, dtype=np.int64).reshape(num_agents)
        for src in range(num_agents):
            for dst in range(num_agents):
                if src == dst:
                    continue
                allowed = (
                    bool(visibility[src, dst])
                    and modes[src] == 2
                    and modes[dst] >= 1
                    and not dead[src]
                    and not dead[dst]
                )
                visibility[src, dst] = allowed

    edge_indices = np.zeros((2, num_agents * num_agents), dtype=np.int64)
    edge_indices[1, :] = -1
    for agent_id in range(num_agents):
        neighbors = np.where(visibility[agent_id])[0]
        start = agent_id * num_agents
        edge_indices[0, start:start + num_agents] = agent_id
        edge_indices[1, start] = agent_id
        edge_indices[1, start + 1:start + 1 + len(neighbors)] = neighbors
    return edge_indices


def scale_policy_action(raw_action: Sequence[float], params: RlControllerParams) -> np.ndarray:
    raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    if raw.shape[0] < 2:
        raise ValueError("RL policy action must contain at least [linear, angular]")
    squashed = np.tanh(raw[:2])
    v = 0.5 * (float(squashed[0]) + 1.0) * max(0.0, params.max_v_mps)
    w = float(squashed[1]) * max(0.0, params.max_w_radps)
    return np.array([v, w], dtype=np.float32)


def safe_half_room(params: RlControllerParams) -> float:
    return max(0.0, 0.5 * params.room_size_m - params.wall_margin_m)


def inside_safe_boundary(pose: Pose2D, params: RlControllerParams) -> bool:
    half = safe_half_room(params)
    return abs(pose.x) <= half and abs(pose.y) <= half


def safety_stop_reason(
    poses: Sequence[Pose2D],
    goals: Sequence[Optional[Pose2D]],
    pose_ages: Sequence[float],
    params: RlControllerParams,
    has_pose: Optional[Sequence[bool]] = None,
    has_goal: Optional[Sequence[bool]] = None,
) -> Optional[str]:
    num_agents = len(params.robot_names)
    if has_pose is None:
        has_pose = [True] * num_agents
    if has_goal is None:
        has_goal = [goal is not None for goal in goals]

    for i, robot in enumerate(params.robot_names):
        if not has_pose[i]:
            return f"missing CV pose for {robot}"
        if pose_ages[i] > params.pose_timeout_s:
            return f"stale CV pose for {robot}"
        if not has_goal[i] or goals[i] is None:
            return f"missing goal for {robot}"
        if not inside_safe_boundary(poses[i], params):
            return f"{robot} pose outside safe boundary"
        if not inside_safe_boundary(goals[i], params):
            return f"{robot} goal outside safe boundary"

    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            d = float(np.linalg.norm(poses[i].xy - poses[j].xy))
            if d < params.min_live_spacing_m:
                return (
                    f"unsafe live spacing between {params.robot_names[i]} and "
                    f"{params.robot_names[j]}: {d:.3f} m"
                )
    return None


def control_to_world(pose: Pose2D, command: Sequence[float]) -> np.ndarray:
    v = float(command[0])
    return np.array([v * math.cos(pose.theta), v * math.sin(pose.theta)], dtype=np.float32)


def clamp_world_velocity(velocity: np.ndarray, params: RlControllerParams) -> np.ndarray:
    out = np.asarray(velocity, dtype=np.float32).copy()
    speed = float(np.linalg.norm(out))
    if speed > params.max_v_mps and speed > 1e-6:
        out *= params.max_v_mps / speed
    return out


def priority_avoidance_shares(
    i: int,
    j: int,
    priorities: Optional[np.ndarray],
    params: RlControllerParams,
) -> Tuple[float, float]:
    share_i = 0.5
    share_j = 0.5
    if params.orca_priority_enabled and priorities is not None:
        priority_diff = float(priorities[j] - priorities[i])
        bias = math.tanh(max(0.0, params.orca_priority_strength) * priority_diff)
        share_i = 0.5 + 0.5 * bias
        min_share = float(np.clip(params.orca_priority_min_share, 0.0, 0.49))
        share_i = float(np.clip(share_i, min_share, 1.0 - min_share))
        share_j = 1.0 - share_i
    return share_i, share_j


def apply_orca_like_filter(
    poses: Sequence[Pose2D],
    world_velocities: np.ndarray,
    params: RlControllerParams,
    priorities: Optional[np.ndarray] = None,
) -> np.ndarray:
    velocities = np.asarray(world_velocities, dtype=np.float32).copy()
    min_dist = max(params.safety_distance_m, 2.0 * params.orca_radius_m)
    horizon = max(0.1, params.orca_time_horizon_s)
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            dx = poses[i].x - poses[j].x
            dy = poses[i].y - poses[j].y
            dist = math.hypot(dx, dy)
            if dist > params.orca_neighbor_dist_m:
                continue
            ux = dx / dist if dist > 1e-6 else 1.0
            uy = dy / dist if dist > 1e-6 else 0.0
            rel_v = velocities[i] - velocities[j]
            closing_speed = -(float(rel_v[0]) * ux + float(rel_v[1]) * uy)
            predicted_dist = dist - max(0.0, closing_speed) * horizon
            if predicted_dist >= min_dist and dist >= min_dist:
                continue
            overlap_term = max(0.0, min_dist - dist) * 2.0
            horizon_term = max(0.0, min_dist - predicted_dist) / horizon
            share_i, share_j = priority_avoidance_shares(i, j, priorities, params)
            correction = params.orca_avoidance_gain * (overlap_term + horizon_term)
            velocities[i, 0] += share_i * correction * ux
            velocities[i, 1] += share_i * correction * uy
            velocities[j, 0] -= share_j * correction * ux
            velocities[j, 1] -= share_j * correction * uy
    return np.asarray([clamp_world_velocity(v, params) for v in velocities], dtype=np.float32)


def apply_boundary_velocity_filter(
    poses: Sequence[Pose2D],
    world_velocities: np.ndarray,
    params: RlControllerParams,
) -> np.ndarray:
    velocities = np.asarray(world_velocities, dtype=np.float32).copy()
    half = safe_half_room(params)
    margin = max(1e-3, params.boundary_slowdown_margin_m)
    for i, pose in enumerate(poses):
        if pose.x > half - margin and velocities[i, 0] > 0.0:
            velocities[i, 0] *= np.clip((half - pose.x) / margin, 0.0, 1.0)
        if pose.x < -half + margin and velocities[i, 0] < 0.0:
            velocities[i, 0] *= np.clip((pose.x + half) / margin, 0.0, 1.0)
        if pose.y > half - margin and velocities[i, 1] > 0.0:
            velocities[i, 1] *= np.clip((half - pose.y) / margin, 0.0, 1.0)
        if pose.y < -half + margin and velocities[i, 1] < 0.0:
            velocities[i, 1] *= np.clip((pose.y + half) / margin, 0.0, 1.0)
        velocities[i] = clamp_world_velocity(velocities[i], params)
    return velocities


def filter_commands_for_safety(
    poses: Sequence[Pose2D],
    commands: np.ndarray,
    params: RlControllerParams,
    priorities: Optional[np.ndarray] = None,
) -> np.ndarray:
    bounded = np.asarray(commands, dtype=np.float32).copy()
    bounded[:, 0] = np.clip(bounded[:, 0], 0.0, params.max_v_mps)
    bounded[:, 1] = np.clip(bounded[:, 1], -params.max_w_radps, params.max_w_radps)
    world_velocities = np.asarray(
        [control_to_world(pose, cmd) for pose, cmd in zip(poses, bounded)],
        dtype=np.float32,
    )
    if params.orca_filter_enabled:
        world_velocities = apply_orca_like_filter(poses, world_velocities, params, priorities)
    world_velocities = apply_boundary_velocity_filter(poses, world_velocities, params)

    filtered = bounded.copy()
    for i, pose in enumerate(poses):
        if bounded[i, 0] <= 1e-6:
            filtered[i, 0] = 0.0
            continue
        forward = np.array([math.cos(pose.theta), math.sin(pose.theta)], dtype=np.float32)
        projected = float(np.dot(world_velocities[i], forward))
        filtered[i, 0] = np.clip(projected, 0.0, bounded[i, 0])
    filtered[:, 1] = np.clip(filtered[:, 1], -params.max_w_radps, params.max_w_radps)
    return filtered.astype(np.float32)


def goal_reached(pose: Pose2D, goal: Optional[Pose2D], params: RlControllerParams) -> bool:
    if goal is None:
        return False
    return float(np.linalg.norm(pose.xy - goal.xy)) <= params.goal_radius_m


def apply_goal_termination_commands(
    poses: Sequence[Pose2D],
    goals: Sequence[Optional[Pose2D]],
    commands: np.ndarray,
    params: RlControllerParams,
) -> np.ndarray:
    out = np.asarray(commands, dtype=np.float32).copy()
    terminal_v = float(np.clip(params.goal_termination_v_mps, 0.0, params.max_v_mps))
    terminal_w = float(
        np.clip(params.goal_termination_w_radps, -params.max_w_radps, params.max_w_radps)
    )
    for i, (pose, goal) in enumerate(zip(poses, goals)):
        if goal_reached(pose, goal, params):
            out[i, 0] = terminal_v
            out[i, 1] = terminal_w
    return out.astype(np.float32)


def slew_limit_commands(
    commands: np.ndarray,
    previous_commands: np.ndarray,
    params: RlControllerParams,
) -> np.ndarray:
    out = np.asarray(commands, dtype=np.float32).copy()
    prev = np.asarray(previous_commands, dtype=np.float32)
    if params.max_dv_step > 0.0:
        out[:, 0] = prev[:, 0] + np.clip(
            out[:, 0] - prev[:, 0],
            -params.max_dv_step,
            params.max_dv_step,
        )
    if params.max_dw_step > 0.0:
        out[:, 1] = prev[:, 1] + np.clip(
            out[:, 1] - prev[:, 1],
            -params.max_dw_step,
            params.max_dw_step,
        )
    out[:, 0] = np.clip(out[:, 0], 0.0, params.max_v_mps)
    out[:, 1] = np.clip(out[:, 1], -params.max_w_radps, params.max_w_radps)
    return out.astype(np.float32)


def validate_aero_marl_root(aero_root: Path) -> Path:
    if not aero_root.exists():
        raise FileNotFoundError(f"AERO-MARL root does not exist: {aero_root}")
    if not aero_root.is_dir():
        raise NotADirectoryError(
            "aero_marl_root must be the AERO-MARL repository directory containing "
            f"mat/config.py, not a checkpoint file: {aero_root}"
        )
    if not (aero_root / "mat" / "config.py").exists():
        raise FileNotFoundError(
            "aero_marl_root must point to the AERO-MARL repository root containing "
            f"mat/config.py; got: {aero_root}"
        )
    return aero_root


class MatPolicyAdapter:
    """Small direct wrapper around AERO-MARL TransformerPolicy."""

    def __init__(self, params: RlControllerParams):
        self.params = params
        self.num_agents = len(params.robot_names)
        self.obs_dim = obs_dim_for_agents(self.num_agents)
        self.state_dim = state_dim_for_agents(self.num_agents)
        self.rnn_states = None

        model_path = Path(params.model_dir).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"RL checkpoint does not exist: {model_path}")
        aero_root = validate_aero_marl_root(Path(params.aero_marl_root).expanduser())
        root_str = str(aero_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        import torch
        from gymnasium import spaces
        from mat.config import get_config

        self.torch = torch
        self._install_torch_scatter_fallback(torch)
        from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy

        self.device = self._select_device(params.policy_device)
        torch.set_num_threads(max(1, int(params.__dict__.get("n_training_threads", 1))))
        args = self._build_policy_args(get_config())

        obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        share_obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )
        action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.policy = TransformerPolicy(
            args,
            obs_space,
            share_obs_space,
            action_space,
            self.num_agents,
            device=self.device,
        )
        self.policy.restore(str(model_path), allow_partial=params.allow_partial_restore)
        self.policy.eval()
        self.recurrent_n = int(getattr(args, "recurrent_N", 1))
        self.n_embd = int(getattr(args, "n_embd", params.n_embd))
        self.reset()

    @staticmethod
    def _install_torch_scatter_fallback(torch_module) -> None:
        try:
            from torch_scatter import scatter_add  # noqa: F401
            return
        except Exception as exc:
            print(
                "[cv_rl_direct] using Python torch_scatter.scatter_add fallback: "
                f"{exc}"
            )

        fallback = types.ModuleType("torch_scatter")

        def scatter_add(src, index, dim=0, out=None, dim_size=None):
            index = index.long()
            if out is None:
                size = list(src.shape)
                if dim_size is None:
                    dim_size = int(index.max().item()) + 1 if index.numel() else 0
                size[dim] = int(dim_size)
                out = torch_module.zeros(size, dtype=src.dtype, device=src.device)
            if index.numel() == 0:
                return out
            return out.index_add_(dim, index, src)

        fallback.scatter_add = scatter_add
        sys.modules["torch_scatter"] = fallback

    def _select_device(self, requested: str):
        requested = (requested or "auto").lower()
        if requested == "auto":
            return self.torch.device("cuda:0" if self.torch.cuda.is_available() else "cpu")
        if requested.startswith("cuda") and not self.torch.cuda.is_available():
            print("[cv_rl_direct] CUDA requested but unavailable; using CPU.")
            return self.torch.device("cpu")
        return self.torch.device(requested)

    def _build_policy_args(self, config_parser_factory):
        parser = config_parser_factory
        args = parser.parse_args([])
        params = self.params
        args.algorithm_name = params.algorithm_name
        args.experiment_name = params.experiment_name
        args.seed = params.seed
        args.env_name = "real_world"
        args.model_dir = params.model_dir
        args.n_training_threads = 1
        args.n_rollout_threads = 1
        args.n_eval_rollout_threads = 1
        args.episode_length = 100
        args.lr = params.lr
        args.critic_lr = params.critic_lr
        args.value_loss_coef = params.value_loss_coef
        args.max_grad_norm = params.max_grad_norm
        args.gamma = params.gamma
        args.gae_lambda = params.gae_lambda
        args.entropy_coef = params.entropy_coef
        args.clip_param = params.clip_param
        args.n_embd = params.n_embd
        args.n_quants = params.n_quants
        args.iterations = params.iterations
        args.num_layers = params.num_layers
        args.num_heads = params.num_heads
        args.n_head = params.num_heads
        args.truelyDistributed = params.truely_distributed
        args.truelyDistributedGNN = params.truely_distributed_gnn
        args.consensusLoss = params.consensus_loss
        args.gnn_loss_coef = params.gnn_loss_coef
        args.sight_range = params.sight_range
        args.use_comm_action = params.use_comm_action
        args.comm_action_dim = params.comm_action_dim
        args.allow_partial_restore = params.allow_partial_restore
        args.clone_extra_agents_from = params.clone_extra_agents_from
        args.use_image_obs = False
        args.use_state_agent = True
        args.add_center_xy = True
        args.num_agents = self.num_agents
        return args

    def reset(self) -> None:
        self.rnn_states = self.torch.zeros(
            (
                1,
                self.num_agents,
                int(getattr(self, "recurrent_n", 1)),
                int(getattr(self, "n_embd", self.params.n_embd)),
            ),
            dtype=self.torch.float32,
            device=self.device,
        )

    @staticmethod
    def _batch_edge_index(edge_index, num_agents, torch_module, device):
        edge_tensor = torch_module.tensor(
            edge_index[None, :, :],
            dtype=torch_module.float32,
            device=device,
        )
        valid = edge_tensor[0, 1, :] != -1
        return edge_tensor[0, :, valid].long()

    def act(
        self,
        obs: np.ndarray,
        share_obs: np.ndarray,
        edge_index: np.ndarray,
    ) -> np.ndarray:
        with self.torch.no_grad():
            obs_t = self.torch.tensor(obs[None, :, :], dtype=self.torch.float32, device=self.device)
            share_t = self.torch.tensor(
                share_obs[None, :, :],
                dtype=self.torch.float32,
                device=self.device,
            )
            if hasattr(self.policy.transformer, "preprocess_obs"):
                obs_t = self.policy.transformer.preprocess_obs(obs_t)

            batch_edge_index = None
            if (
                self.params.iterations > 0
                and self.params.algorithm_name
                in ("mappo_gnn", "mappo_dgnn", "mappo_dgnn_dsgd")
            ):
                batch_edge_index = self._batch_edge_index(
                    edge_index,
                    self.num_agents,
                    self.torch,
                    self.device,
                )
                if hasattr(self.policy.transformer, "gnn_input"):
                    gnn_obs = self.policy.transformer.gnn_input(obs_t)
                else:
                    gnn_obs = obs_t
                encoded = self.policy.transformer.obs_encoder(gnn_obs, batch_edge_index)
                obs_t = self.torch.cat([obs_t, encoded], dim=-1).detach()

            masks = self.torch.ones(
                (1, self.num_agents, 1),
                dtype=self.torch.float32,
                device=self.device,
            )
            actions, self.rnn_states = self.policy.act(
                share_t,
                obs_t,
                self.rnn_states,
                masks,
                None,
                deterministic=True,
                batched_edge_index=batch_edge_index,
            )
            return actions.reshape(self.num_agents, -1).detach().cpu().numpy()


class CVRlDirectController(Node):
    def __init__(self):
        if rclpy is None:
            raise ImportError("rclpy and ROS 2 message packages are required.")
        super().__init__("cv_rl_direct_controller")
        self.params = self._declare_and_load_params()
        self.robot_names = self.params.robot_names
        self.active = False
        self.comm_modes = np.full(len(self.robot_names), 2, dtype=np.int64)
        self.last_status = ""
        self.last_status_time = 0.0
        self.last_velocity_status_time = 0.0
        self.robots: Dict[str, RobotRuntime] = {
            robot: RobotRuntime() for robot in self.robot_names
        }
        self._rl_subscription_refs = []

        self.status_pub = self.create_publisher(String, "/fleet_mppi/status", 10)
        self.velocity_status_pub = self.create_publisher(
            String,
            "/fleet_mppi/velocity_status",
            10,
        )

        for robot in self.robot_names:
            runtime = self.robots[robot]
            runtime.cmd_pub = self.create_publisher(Twist, f"/{robot}/cmd_vel", 10)
            runtime.measured_velocity_pub = self.create_publisher(
                Twist,
                f"/{robot}/cv_measured_velocity",
                10,
            )
            runtime.path_pub = self.create_publisher(RosPath, f"/{robot}/mppi_plan", 10)
            self._rl_subscription_refs.append(
                self.create_subscription(
                    PoseStamped,
                    f"/{robot}/cv_pose",
                    lambda msg, name=robot: self._pose_cb(name, msg),
                    10,
                )
            )
            self._rl_subscription_refs.append(
                self.create_subscription(
                    Odometry,
                    f"/{robot}/odom",
                    lambda msg, name=robot: self._odom_cb(name, msg),
                    10,
                )
            )
            self._rl_subscription_refs.append(
                self.create_subscription(
                    PointStamped,
                    f"/{robot}/mppi_goal",
                    lambda msg, name=robot: self._goal_cb(name, msg),
                    10,
                )
            )

        self.start_srv = self.create_service(Trigger, "/fleet_mppi/start", self._start_srv)
        self.stop_srv = self.create_service(Trigger, "/fleet_mppi/stop", self._stop_srv)
        self.clear_srv = self.create_service(
            Trigger,
            "/fleet_mppi/clear_goals",
            self._clear_srv,
        )
        self.plan_srv = self.create_service(Trigger, "/fleet_mppi/plan", self._plan_srv)

        self.publish_status("loading RL checkpoint", True)
        self.policy = MatPolicyAdapter(self.params)
        self.publish_status(
            "RL direct controller ready; limits v<="
            f"{self.params.max_v_mps:.3f} m/s, w<={self.params.max_w_radps:.3f} rad/s",
            True,
        )

        rate = max(1.0, self.params.control_rate_hz)
        self.timer = self.create_timer(1.0 / rate, self._timer_cb)

    def _declare_and_load_params(self) -> RlControllerParams:
        declare = self.declare_parameter
        declare("robot_names_csv", "tb_1,tb_2,tb_3")
        declare("map_frame", "map")
        declare("control_rate_hz", 20.0)
        declare("pose_timeout_s", 0.5)
        declare("min_live_spacing_m", 0.25)
        declare("room_size_m", 3.048)
        declare("goal_radius_m", 0.12)
        declare("goal_termination_v_mps", 0.0)
        declare("goal_termination_w_radps", 0.0)
        declare("max_v_mps", 0.1)
        declare("max_w_radps", 1.0)
        declare("max_dv_step", 0.03)
        declare("max_dw_step", 0.18)
        declare("safety_distance_m", 0.25)
        declare("wall_margin_m", 0.20)
        declare("boundary_slowdown_margin_m", 0.15)
        declare("orca_filter_enabled", True)
        declare("orca_radius_m", 0.20)
        declare("orca_neighbor_dist_m", 1.5)
        declare("orca_time_horizon_s", 3.0)
        declare("orca_avoidance_gain", 3.0)
        declare("orca_priority_enabled", True)
        declare("orca_priority_strength", 5.0)
        declare("orca_priority_fixed_bias", 0.25)
        declare("orca_priority_min_share", 0.10)
        declare("sight_range", 9.0)
        declare("model_dir", "")
        declare("aero_marl_root", "/home/adi2440/Desktop/MARL_Shahil_Aditya/AERO-MARL")
        declare("policy_device", "auto")
        declare("allow_partial_restore", True)
        declare("clone_extra_agents_from", 0)
        declare("use_comm_action", True)

        get = lambda name: self.get_parameter(name).value
        model_dir = str(get("model_dir")).strip()
        if not model_dir:
            raise RuntimeError("model_dir is required for cv_rl_direct_controller")
        robot_names = split_csv(str(get("robot_names_csv")))
        if not robot_names:
            raise RuntimeError("robot_names_csv must contain at least one robot")
        return RlControllerParams(
            robot_names=robot_names,
            map_frame=str(get("map_frame")),
            control_rate_hz=float(get("control_rate_hz")),
            pose_timeout_s=float(get("pose_timeout_s")),
            min_live_spacing_m=float(get("min_live_spacing_m")),
            room_size_m=float(get("room_size_m")),
            goal_radius_m=float(get("goal_radius_m")),
            goal_termination_v_mps=float(get("goal_termination_v_mps")),
            goal_termination_w_radps=float(get("goal_termination_w_radps")),
            max_v_mps=float(get("max_v_mps")),
            max_w_radps=float(get("max_w_radps")),
            max_dv_step=float(get("max_dv_step")),
            max_dw_step=float(get("max_dw_step")),
            safety_distance_m=float(get("safety_distance_m")),
            wall_margin_m=float(get("wall_margin_m")),
            boundary_slowdown_margin_m=float(get("boundary_slowdown_margin_m")),
            orca_filter_enabled=bool(get("orca_filter_enabled")),
            orca_radius_m=float(get("orca_radius_m")),
            orca_neighbor_dist_m=float(get("orca_neighbor_dist_m")),
            orca_time_horizon_s=float(get("orca_time_horizon_s")),
            orca_avoidance_gain=float(get("orca_avoidance_gain")),
            orca_priority_enabled=bool(get("orca_priority_enabled")),
            orca_priority_strength=float(get("orca_priority_strength")),
            orca_priority_fixed_bias=float(get("orca_priority_fixed_bias")),
            orca_priority_min_share=float(get("orca_priority_min_share")),
            sight_range=float(get("sight_range")),
            model_dir=model_dir,
            aero_marl_root=str(get("aero_marl_root")),
            policy_device=str(get("policy_device")),
            allow_partial_restore=bool(get("allow_partial_restore")),
            clone_extra_agents_from=int(get("clone_extra_agents_from")),
            use_comm_action=bool(get("use_comm_action")),
        )

    def _pose_cb(self, robot: str, msg: PoseStamped) -> None:
        runtime = self.robots[robot]
        now = time.monotonic()
        if runtime.has_pose:
            runtime.previous_pose = runtime.pose
            runtime.previous_pose_time = runtime.pose_time
        q = msg.pose.orientation
        runtime.pose = Pose2D(
            x=float(msg.pose.position.x),
            y=float(msg.pose.position.y),
            theta=yaw_from_quat_xyzw(q.x, q.y, q.z, q.w),
        )
        runtime.pose_time = now
        runtime.has_pose = True
        if runtime.previous_pose is not None:
            dt = max(1e-6, now - runtime.previous_pose_time)
            dx = runtime.pose.x - runtime.previous_pose.x
            dy = runtime.pose.y - runtime.previous_pose.y
            vx_world = dx / dt
            vy_world = dy / dt
            heading = runtime.pose.theta
            runtime.measured_cv_velocity[0] = (
                vx_world * math.cos(heading) + vy_world * math.sin(heading)
            )
            runtime.measured_cv_lateral_mps = (
                -vx_world * math.sin(heading) + vy_world * math.cos(heading)
            )
            runtime.measured_cv_speed_mps = math.hypot(vx_world, vy_world)
            runtime.measured_cv_velocity[1] = (
                wrap_to_pi(runtime.pose.theta - runtime.previous_pose.theta) / dt
            )
            self.publish_measured_velocity(runtime)

    def _odom_cb(self, robot: str, msg: Odometry) -> None:
        runtime = self.robots[robot]
        runtime.measured_odom_velocity[0] = float(msg.twist.twist.linear.x)
        runtime.measured_odom_velocity[1] = float(msg.twist.twist.angular.z)
        runtime.has_odom_velocity = True

    def _goal_cb(self, robot: str, msg) -> None:
        if msg.header.frame_id and msg.header.frame_id != self.params.map_frame:
            self.publish_status(
                f"accepted {robot} goal without TF transform; expected frame "
                f"{self.params.map_frame}",
                True,
            )
        goal = Pose2D(float(msg.point.x), float(msg.point.y), 0.0)
        if not inside_safe_boundary(goal, self.params):
            self.robots[robot].has_goal = False
            self.publish_status(f"rejected {robot} goal: outside safe boundary", True)
            return
        runtime = self.robots[robot]
        runtime.goal = goal
        runtime.has_goal = True
        runtime.priority_wait_s = 0.0
        self.publish_status(
            f"stored {robot} RL goal x={goal.x:.3f} y={goal.y:.3f}",
            True,
        )
        self.publish_plan_previews()

    def _start_srv(self, _request, response):
        reason = self.ready_stop_reason()
        if reason is not None:
            response.success = False
            response.message = reason
            self.publish_status(f"cannot start RL control: {reason}", True)
            return response
        self.policy.reset()
        self.active = True
        response.success = True
        response.message = "RL checkpoint control started"
        self.publish_status(response.message, True)
        return response

    def _stop_srv(self, _request, response):
        self.stop_all("stop service")
        response.success = True
        response.message = "stopped all robots"
        return response

    def _clear_srv(self, _request, response):
        self.stop_all("clear goals")
        for robot in self.robot_names:
            runtime = self.robots[robot]
            runtime.goal = None
            runtime.has_goal = False
            runtime.priority_wait_s = 0.0
            self.publish_path(robot, [])
        response.success = True
        response.message = "cleared RL goals"
        self.publish_status(response.message, True)
        return response

    def _plan_srv(self, _request, response):
        reason = self.ready_stop_reason(require_spacing=False)
        if reason is not None:
            response.success = False
            response.message = reason
            self.publish_status(f"cannot preview RL run: {reason}", True)
            return response
        self.publish_plan_previews()
        response.success = True
        response.message = "RL checkpoint preview ready"
        self.publish_status(response.message, True)
        return response

    def ready_stop_reason(self, require_spacing: bool = True) -> Optional[str]:
        now = time.monotonic()
        poses = [self.robots[robot].pose for robot in self.robot_names]
        goals = [self.robots[robot].goal for robot in self.robot_names]
        pose_ages = [now - self.robots[robot].pose_time for robot in self.robot_names]
        has_pose = [self.robots[robot].has_pose for robot in self.robot_names]
        has_goal = [self.robots[robot].has_goal for robot in self.robot_names]
        reason = safety_stop_reason(
            poses,
            goals,
            pose_ages,
            self.params,
            has_pose=has_pose,
            has_goal=has_goal,
        )
        if reason and not require_spacing and reason.startswith("unsafe live spacing"):
            return None
        return reason

    def all_goals_ready(self) -> bool:
        return all(self.robots[robot].has_goal for robot in self.robot_names)

    def final_goals_reached(self) -> bool:
        for robot in self.robot_names:
            runtime = self.robots[robot]
            if runtime.goal is None:
                return False
            if np.linalg.norm(runtime.pose.xy - runtime.goal.xy) > self.params.goal_radius_m:
                return False
        return True

    def _timer_cb(self) -> None:
        if not self.all_goals_ready():
            return
        reason = self.ready_stop_reason()
        if reason is not None:
            if self.active:
                self.stop_all(reason)
            else:
                self.publish_status(f"waiting for RL preview: {reason}")
            return
        self.publish_plan_previews()
        if self.final_goals_reached():
            if self.active:
                self.stop_all("all goals reached")
            self.publish_status("all RL goals reached")
            return
        if not self.active:
            self.publish_status("RL checkpoint preview ready")
            return

        poses = [self.robots[robot].pose for robot in self.robot_names]
        goals = [self.robots[robot].goal for robot in self.robot_names]
        previous = np.asarray(
            [self.robots[robot].last_cmd for robot in self.robot_names],
            dtype=np.float32,
        )
        obs, share_obs, edge_index = build_policy_inputs(
            poses,
            goals,
            previous,
            self.params,
            self.comm_modes,
        )
        raw_actions = self.policy.act(obs, share_obs, edge_index)
        motion_actions = raw_actions[:, :2]
        if self.params.use_comm_action and raw_actions.shape[1] > 2:
            self.comm_modes = np.clip(np.rint(raw_actions[:, 2]), 0, 2).astype(np.int64)
        commands = np.asarray(
            [scale_policy_action(action, self.params) for action in motion_actions],
            dtype=np.float32,
        )
        commands = apply_goal_termination_commands(poses, goals, commands, self.params)
        self.update_priority_wait(commands)
        priorities = self.compute_priorities()
        commands = filter_commands_for_safety(poses, commands, self.params, priorities)
        commands = apply_goal_termination_commands(poses, goals, commands, self.params)
        commands = slew_limit_commands(commands, previous, self.params)
        commands = apply_goal_termination_commands(poses, goals, commands, self.params)
        for i, robot in enumerate(self.robot_names):
            self.publish_control(robot, commands[i])
        self.publish_velocity_status()

    def update_priority_wait(self, commands: np.ndarray) -> None:
        dt = 1.0 / max(1.0, self.params.control_rate_hz)
        for i, robot in enumerate(self.robot_names):
            runtime = self.robots[robot]
            wants_motion = abs(float(commands[i, 0])) > 0.01
            moving_slowly = runtime.measured_cv_speed_mps < 0.02
            if wants_motion and moving_slowly:
                runtime.priority_wait_s += dt
            else:
                runtime.priority_wait_s = max(0.0, runtime.priority_wait_s - 2.0 * dt)

    def compute_priorities(self) -> np.ndarray:
        priorities = []
        for i, robot in enumerate(self.robot_names):
            deterministic_bias = self.params.orca_priority_fixed_bias * (
                len(self.robot_names) - 1 - i
            )
            priorities.append(deterministic_bias + self.robots[robot].priority_wait_s)
        return np.asarray(priorities, dtype=np.float32)

    def publish_control(self, robot: str, command: Sequence[float]) -> None:
        runtime = self.robots[robot]
        bounded = np.asarray(command, dtype=np.float32).copy()
        bounded[0] = np.clip(bounded[0], 0.0, self.params.max_v_mps)
        bounded[1] = np.clip(bounded[1], -self.params.max_w_radps, self.params.max_w_radps)
        msg = Twist()
        msg.linear.x = float(bounded[0])
        msg.angular.z = float(bounded[1])
        runtime.cmd_pub.publish(msg)
        runtime.last_cmd[:] = bounded

    def publish_measured_velocity(self, runtime: RobotRuntime) -> None:
        if runtime.measured_velocity_pub is None:
            return
        msg = Twist()
        msg.linear.x = float(runtime.measured_cv_velocity[0])
        msg.linear.y = float(runtime.measured_cv_lateral_mps)
        msg.linear.z = float(runtime.measured_cv_speed_mps)
        msg.angular.z = float(runtime.measured_cv_velocity[1])
        runtime.measured_velocity_pub.publish(msg)

    def publish_zero_burst(self) -> None:
        if rclpy is not None and not rclpy.ok():
            return
        zero = Twist()
        for _ in range(5):
            for robot in self.robot_names:
                try:
                    self.robots[robot].cmd_pub.publish(zero)
                except Exception:
                    return

    def stop_all(self, reason: str) -> None:
        self.active = False
        for robot in self.robot_names:
            runtime = self.robots[robot]
            runtime.last_cmd[:] = 0.0
            runtime.priority_wait_s = 0.0
        self.publish_zero_burst()
        self.publish_status(f"stopped all robots: {reason}", True)

    def publish_plan_previews(self) -> None:
        for robot in self.robot_names:
            runtime = self.robots[robot]
            if runtime.has_pose and runtime.has_goal and runtime.goal is not None:
                self.publish_path(robot, [runtime.pose, runtime.goal])
            else:
                self.publish_path(robot, [])

    def publish_path(self, robot: str, path_states: Sequence[Pose2D]) -> None:
        path = RosPath()
        path.header.frame_id = self.params.map_frame
        path.header.stamp = self.get_clock().now().to_msg()
        if len(path_states) >= 2:
            start, goal = path_states[0], path_states[-1]
            dist = float(np.linalg.norm(start.xy - goal.xy))
            steps = max(2, int(dist / 0.05) + 1)
            expanded = [
                Pose2D(
                    x=start.x + (goal.x - start.x) * t / (steps - 1),
                    y=start.y + (goal.y - start.y) * t / (steps - 1),
                    theta=0.0,
                )
                for t in range(steps)
            ]
        else:
            expanded = list(path_states)
        for state in expanded:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(state.x)
            pose.pose.position.y = float(state.y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.robots[robot].path_pub.publish(path)

    def publish_velocity_status(self) -> None:
        now = time.monotonic()
        if now - self.last_velocity_status_time < 1.0:
            return
        self.last_velocity_status_time = now
        parts = []
        for robot in self.robot_names:
            runtime = self.robots[robot]
            part = (
                f"{robot} cmd=({runtime.last_cmd[0]:.3f} m/s,"
                f"{runtime.last_cmd[1]:.3f} rad/s) "
                f"cv_fwd=({runtime.measured_cv_velocity[0]:.3f} m/s,"
                f"{runtime.measured_cv_velocity[1]:.3f} rad/s) "
                f"cv_speed={runtime.measured_cv_speed_mps:.3f} m/s"
            )
            if runtime.has_odom_velocity:
                part += (
                    f" odom=({runtime.measured_odom_velocity[0]:.3f} m/s,"
                    f"{runtime.measured_odom_velocity[1]:.3f} rad/s)"
                )
            parts.append(part)
        msg = String()
        msg.data = " | ".join(parts)
        self.velocity_status_pub.publish(msg)
        self.get_logger().info(msg.data)

    def publish_status(self, text: str, force: bool = False) -> None:
        now = time.monotonic()
        if not force and text == self.last_status and now - self.last_status_time < 2.0:
            return
        self.last_status = text
        self.last_status_time = now
        if String is None:
            return
        msg = String()
        msg.data = text
        try:
            self.status_pub.publish(msg)
            self.get_logger().info(text)
        except Exception:
            return

    def stop_for_shutdown(self) -> None:
        if rclpy is not None and not rclpy.ok():
            return
        self.stop_all("shutdown")


def main(argv: Optional[Iterable[str]] = None) -> None:
    if rclpy is None:
        raise RuntimeError("ROS 2 Python packages are not available.")
    args = list(argv) if argv is not None else sys.argv
    rclpy.init(args=args)
    node = None
    try:
        node = CVRlDirectController()
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
