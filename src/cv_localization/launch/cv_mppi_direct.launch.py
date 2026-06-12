import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml


def _sanitized_ld_library_path():
    required_paths = [
        '/opt/ros/humble/lib',
        '/opt/ros/humble/lib/x86_64-linux-gnu',
        '/usr/lib/x86_64-linux-gnu',
        '/usr/local/cuda-13.0/lib64',
    ]
    current_paths = [
        path for path in os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep)
        if path and '{' not in path and 'anaconda3' not in path
    ]
    return os.pathsep.join(dict.fromkeys(required_paths + current_paths))


def _mppi_params_from_config(config_path, robot_names):
    with open(config_path) as stream:
        cfg = yaml.safe_load(stream) or {}

    workspace_cfg = cfg.get('workspace', {})
    mppi_cfg = cfg.get('mppi', {})
    params = {
        'robot_names_csv': robot_names,
        'map_frame': cfg.get('ros', {}).get('global_frame', 'map'),
        'control_rate_hz': float(mppi_cfg.get('control_rate_hz', 10.0)),
        'pose_timeout_s': float(mppi_cfg.get('pose_timeout_s', 0.5)),
        'min_live_spacing_m': float(mppi_cfg.get('min_live_spacing_m', 0.22)),
        'control_mode': str(mppi_cfg.get('control_mode', 'scheduled')),
        'planner_algorithm': str(mppi_cfg.get('planner_algorithm', 'cbs')),
        'dt': float(mppi_cfg.get('dt', 0.10)),
        'room_size_m': float(mppi_cfg.get('room_size_m', workspace_cfg.get('width_m', 3.048))),
        'goal_radius_m': float(mppi_cfg.get('goal_radius_m', 0.12)),
        'max_v_mps': float(mppi_cfg.get('max_v_mps', 0.50)),
        'allow_reverse': bool(mppi_cfg.get('allow_reverse', True)),
        'max_reverse_v_mps': float(
            mppi_cfg.get('max_reverse_v_mps', mppi_cfg.get('max_v_mps', 0.50))),
        'max_w_radps': float(mppi_cfg.get('max_w_radps', 0.20)),
        'max_dv_step': float(mppi_cfg.get('max_dv_step', 0.03)),
        'max_dw_step': float(mppi_cfg.get('max_dw_step', 0.18)),
        'safety_distance_m': float(mppi_cfg.get('safety_distance_m', 0.25)),
        'planning_clearance_m': float(mppi_cfg.get('planning_clearance_m', 0.45)),
        'wall_margin_m': float(mppi_cfg.get('wall_margin_m', 0.20)),
        'horizon': int(mppi_cfg.get('horizon', 80)),
        'samples': int(mppi_cfg.get('samples', 512)),
        'mppi_iterations': int(mppi_cfg.get('mppi_iterations', 6)),
        'offline_planner_enabled': bool(mppi_cfg.get('offline_planner_enabled', True)),
        'offline_grid_resolution_m': float(mppi_cfg.get('offline_grid_resolution_m', 0.10)),
        'offline_time_step_s': float(mppi_cfg.get('offline_time_step_s', 0.50)),
        'offline_max_time_steps': int(mppi_cfg.get('offline_max_time_steps', 180)),
        'offline_wait_penalty': float(mppi_cfg.get('offline_wait_penalty', 0.05)),
        'cbs_max_nodes': int(mppi_cfg.get('cbs_max_nodes', 512)),
        'waypoint_reached_m': float(mppi_cfg.get('waypoint_reached_m', 0.14)),
        'waypoint_lookahead_m': float(mppi_cfg.get('waypoint_lookahead_m', 0.30)),
        'path_heading_gain': float(mppi_cfg.get('path_heading_gain', 2.0)),
        'reverse_heading_threshold_rad': float(
            mppi_cfg.get('reverse_heading_threshold_rad', 2.20)),
        'path_slow_heading_rad': float(mppi_cfg.get('path_slow_heading_rad', 0.70)),
        'path_stop_heading_rad': float(mppi_cfg.get('path_stop_heading_rad', 1.40)),
        'path_goal_slowdown_m': float(mppi_cfg.get('path_goal_slowdown_m', 0.35)),
        'orca_filter_enabled': bool(mppi_cfg.get('orca_filter_enabled', True)),
        'orca_radius_m': float(mppi_cfg.get('orca_radius_m', 0.16)),
        'orca_neighbor_dist_m': float(mppi_cfg.get('orca_neighbor_dist_m', 0.75)),
        'orca_time_horizon_s': float(mppi_cfg.get('orca_time_horizon_s', 2.0)),
        'orca_avoidance_gain': float(mppi_cfg.get('orca_avoidance_gain', 1.0)),
        'orca_priority_enabled': bool(mppi_cfg.get('orca_priority_enabled', True)),
        'orca_priority_strength': float(mppi_cfg.get('orca_priority_strength', 2.0)),
        'orca_priority_fixed_bias': float(mppi_cfg.get('orca_priority_fixed_bias', 0.25)),
        'orca_priority_wait_gain': float(mppi_cfg.get('orca_priority_wait_gain', 0.6)),
        'orca_priority_schedule_lag_gain': float(
            mppi_cfg.get('orca_priority_schedule_lag_gain', 0.2)),
        'orca_priority_min_share': float(mppi_cfg.get('orca_priority_min_share', 0.10)),
        'orca_priority_wait_speed_mps': float(
            mppi_cfg.get('orca_priority_wait_speed_mps', 0.02)),
        'orca_priority_wait_decay_gain': float(
            mppi_cfg.get('orca_priority_wait_decay_gain', 2.0)),
        'boundary_slowdown_margin_m': float(mppi_cfg.get('boundary_slowdown_margin_m', 0.15)),
        'crossing_gate_enabled': bool(mppi_cfg.get('crossing_gate_enabled', False)),
        'crossing_center_x_m': float(mppi_cfg.get('crossing_center_x_m', 0.0)),
        'crossing_center_y_m': float(mppi_cfg.get('crossing_center_y_m', 0.0)),
        'crossing_zone_radius_m': float(mppi_cfg.get('crossing_zone_radius_m', 0.36)),
        'crossing_entry_radius_m': float(mppi_cfg.get('crossing_entry_radius_m', 0.62)),
        'crossing_release_radius_m': float(mppi_cfg.get('crossing_release_radius_m', 0.50)),
        'crossing_progress_timeout_s': float(mppi_cfg.get('crossing_progress_timeout_s', 4.0)),
        'crossing_progress_epsilon_m': float(mppi_cfg.get('crossing_progress_epsilon_m', 0.03)),
        'velocity_status_period_s': float(mppi_cfg.get('velocity_status_period_s', 1.0)),
    }
    return params


def _launch_nodes(context):
    robot_names = LaunchConfiguration('robot_names').perform(context)
    config = LaunchConfiguration('config').perform(context)
    calibration = LaunchConfiguration('calibration').perform(context)
    background = LaunchConfiguration('background').perform(context)

    return [
        Node(
            package='multi_robot_swarm_planner',
            executable='mppi_direct_controller',
            name='mppi_direct_controller',
            output='screen',
            parameters=[_mppi_params_from_config(config, robot_names)],
        ),
        Node(
            package='cv_localization',
            executable='cv_mppi_direct_gui',
            name='cv_mppi_direct_gui',
            output='screen',
            arguments=[
                '--config', config,
                '--calibration', calibration,
                '--background', background,
            ],
        ),
    ]


def generate_launch_description():
    share_dir = get_package_share_directory('cv_localization')
    config_dir = os.path.join(share_dir, 'config')

    return LaunchDescription([
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger'),
        SetEnvironmentVariable('LD_LIBRARY_PATH', _sanitized_ld_library_path()),
        DeclareLaunchArgument(
            'robot_names',
            default_value='tb_1,tb_2,tb_3',
            description='Comma-separated robot namespaces controlled by direct MPPI.',
        ),
        DeclareLaunchArgument(
            'config',
            default_value=os.path.join(config_dir, 'config.yaml'),
            description='CV detector/camera configuration YAML.',
        ),
        DeclareLaunchArgument(
            'calibration',
            default_value=os.path.join(config_dir, 'calibration.yaml'),
            description='Workspace homography calibration YAML.',
        ),
        DeclareLaunchArgument(
            'background',
            default_value=os.path.join(config_dir, 'background.jpg'),
            description='Fixed one-time background image.',
        ),
        OpaqueFunction(function=_launch_nodes),
    ])
