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


def _load_config(config_path):
    with open(config_path) as stream:
        return yaml.safe_load(stream) or {}


def _resolve_collection_sample_rate(config_path, sample_rate_hz):
    value = str(sample_rate_hz).strip()
    if value.lower() in ('', 'from_config', 'config'):
        cfg = _load_config(config_path)
        collection_cfg = cfg.get('vlcm_collection', {})
        return float(collection_cfg.get('sample_rate_hz', 30.0))
    return float(value)


def _rl_params_from_config(config_path, robot_names, model_dir, aero_marl_root):
    cfg = _load_config(config_path)
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
        'goal_termination_v_mps': float(mppi_cfg.get('goal_termination_v_mps', 0.0)),
        'goal_termination_w_radps': float(mppi_cfg.get('goal_termination_w_radps', 0.0)),
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


def _collector_params_from_config(
    config_path,
    robot_names,
    episodes,
    dataset_dir,
    sample_rate_hz,
    random_seed,
    goal_min_separation_m,
    goal_min_robot_distance_m,
):
    cfg = _load_config(config_path)
    workspace_cfg = cfg.get('workspace', {})
    mppi_cfg = cfg.get('mppi', {})
    ros_cfg = cfg.get('ros', {})
    return {
        'robot_names_csv': robot_names,
        'map_frame': ros_cfg.get('global_frame', 'map'),
        'episodes': int(episodes),
        'dataset_dir': dataset_dir,
        'sample_rate_hz': float(sample_rate_hz),
        'random_seed': int(random_seed),
        'workspace_width_m': float(workspace_cfg.get('width_m', 3.048)),
        'workspace_height_m': float(workspace_cfg.get('height_m', 3.048)),
        'wall_margin_m': float(mppi_cfg.get('wall_margin_m', ros_cfg.get('boundary_margin_m', 0.20))),
        'goal_radius_m': float(mppi_cfg.get('goal_radius_m', 0.12)),
        'goal_min_separation_m': float(goal_min_separation_m),
        'goal_min_robot_distance_m': float(goal_min_robot_distance_m),
    }


def _launch_nodes(context):
    robot_names = LaunchConfiguration('robot_names').perform(context)
    config = LaunchConfiguration('config').perform(context)
    calibration = LaunchConfiguration('calibration').perform(context)
    background = LaunchConfiguration('background').perform(context)
    model_dir = LaunchConfiguration('model_dir').perform(context)
    aero_marl_root = LaunchConfiguration('aero_marl_root').perform(context)
    episodes = LaunchConfiguration('episodes').perform(context)
    dataset_dir = LaunchConfiguration('dataset_dir').perform(context)
    sample_rate_hz = _resolve_collection_sample_rate(
        config,
        LaunchConfiguration('sample_rate_hz').perform(context),
    )
    random_seed = LaunchConfiguration('random_seed').perform(context)
    goal_min_separation_m = LaunchConfiguration('goal_min_separation_m').perform(context)
    goal_min_robot_distance_m = LaunchConfiguration('goal_min_robot_distance_m').perform(context)

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
            parameters=[{'vlcm_frame_publish_hz': float(sample_rate_hz)}],
        ),
        Node(
            package='cv_localization',
            executable='tb3_vlcm_live_collector',
            name='tb3_vlcm_live_collector',
            output='screen',
            parameters=[_collector_params_from_config(
                config,
                robot_names,
                episodes,
                dataset_dir,
                sample_rate_hz,
                random_seed,
                goal_min_separation_m,
                goal_min_robot_distance_m,
            )],
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
            default_value='/home/adi2440/turtlebot_ws/models/transformer_800.pt',
            description='AERO-MARL transformer checkpoint path. Must exist before launch.',
        ),
        DeclareLaunchArgument(
            'aero_marl_root',
            default_value='/home/adi2440/Desktop/MARL_Shahil_Aditya/AERO-MARL',
            description='AERO-MARL repository root used for importing mat.* modules.',
        ),
        DeclareLaunchArgument(
            'episodes',
            default_value='100',
            description='Number of accepted TurtleBot lab episodes to collect.',
        ),
        DeclareLaunchArgument(
            'dataset_dir',
            default_value='/home/adi2440/Desktop/MARL_Shahil_Aditya/MA-VLCM/data/tb3_lab',
            description='Directory where per-episode MA-VLCM WebDataset shards are written.',
        ),
        DeclareLaunchArgument(
            'sample_rate_hz',
            default_value='from_config',
            description=(
                'Frame/state sampling rate for MA-VLCM data collection, or '
                '"from_config" to use vlcm_collection.sample_rate_hz.'
            ),
        ),
        DeclareLaunchArgument(
            'random_seed',
            default_value='0',
            description='Random seed for goal generation; 0 uses system randomness.',
        ),
        DeclareLaunchArgument(
            'goal_min_separation_m',
            default_value='0.5',
            description='Minimum distance between generated goals.',
        ),
        DeclareLaunchArgument(
            'goal_min_robot_distance_m',
            default_value='0.35',
            description='Minimum distance from each robot current pose to its generated goal.',
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
