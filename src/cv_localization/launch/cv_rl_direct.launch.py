import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable, Shutdown
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


def _rl_params_from_config(config_path, robot_names, model_dir, aero_marl_root):
    with open(config_path) as stream:
        cfg = yaml.safe_load(stream) or {}

    workspace_cfg = cfg.get('workspace', {})
    mppi_cfg = cfg.get('mppi', {})
    return {
        'robot_names_csv': robot_names,
        'map_frame': cfg.get('ros', {}).get('global_frame', 'map'),
        'control_rate_hz': float(mppi_cfg.get('control_rate_hz', 20.0)),
        'pose_timeout_s': float(mppi_cfg.get('pose_timeout_s', 0.5)),
        'min_live_spacing_m': float(mppi_cfg.get('min_live_spacing_m', 0.25)),
        'room_size_m': float(mppi_cfg.get('room_size_m', workspace_cfg.get('width_m', 3.048))),
        'goal_radius_m': float(mppi_cfg.get('goal_radius_m', 0.12)),
        'max_v_mps': float(mppi_cfg.get('max_v_mps', 0.10)),
        'max_w_radps': float(mppi_cfg.get('max_w_radps', 1.0)),
        'max_dv_step': float(mppi_cfg.get('max_dv_step', 0.03)),
        'max_dw_step': float(mppi_cfg.get('max_dw_step', 0.18)),
        'safety_distance_m': float(mppi_cfg.get('safety_distance_m', 0.25)),
        'wall_margin_m': float(mppi_cfg.get('wall_margin_m', 0.20)),
        'boundary_slowdown_margin_m': float(mppi_cfg.get('boundary_slowdown_margin_m', 0.15)),
        'orca_filter_enabled': bool(mppi_cfg.get('orca_filter_enabled', True)),
        'orca_radius_m': float(mppi_cfg.get('orca_radius_m', 0.20)),
        'orca_neighbor_dist_m': float(mppi_cfg.get('orca_neighbor_dist_m', 1.5)),
        'orca_time_horizon_s': float(mppi_cfg.get('orca_time_horizon_s', 3.0)),
        'orca_avoidance_gain': float(mppi_cfg.get('orca_avoidance_gain', 3.0)),
        'orca_priority_enabled': bool(mppi_cfg.get('orca_priority_enabled', True)),
        'orca_priority_strength': float(mppi_cfg.get('orca_priority_strength', 5.0)),
        'orca_priority_fixed_bias': float(mppi_cfg.get('orca_priority_fixed_bias', 0.25)),
        'orca_priority_min_share': float(mppi_cfg.get('orca_priority_min_share', 0.10)),
        'model_dir': model_dir,
        'aero_marl_root': aero_marl_root,
    }


def _launch_nodes(context):
    robot_names = LaunchConfiguration('robot_names').perform(context)
    config = LaunchConfiguration('config').perform(context)
    calibration = LaunchConfiguration('calibration').perform(context)
    background = LaunchConfiguration('background').perform(context)
    model_dir = LaunchConfiguration('model_dir').perform(context)
    aero_marl_root = LaunchConfiguration('aero_marl_root').perform(context)

    return [
        Node(
            package='cv_localization',
            executable='cv_rl_direct_controller',
            name='cv_rl_direct_controller',
            output='screen',
            parameters=[_rl_params_from_config(config, robot_names, model_dir, aero_marl_root)],
            on_exit=[Shutdown(reason='cv_rl_direct_controller exited')],
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
            description='Comma-separated robot namespaces controlled by the RL checkpoint.',
        ),
        DeclareLaunchArgument(
            'model_dir',
            default_value='/home/i2r/shahil_ws/AERO-MARL/transformer_780.pt',
            description='AERO-MARL transformer checkpoint path. Must exist before launch.',
        ),
        DeclareLaunchArgument(
            'aero_marl_root',
            default_value='/home/adi2440/Desktop/MARL_Shahil_Aditya/AERO-MARL',
            description='AERO-MARL repository root used for importing mat.* modules.',
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
