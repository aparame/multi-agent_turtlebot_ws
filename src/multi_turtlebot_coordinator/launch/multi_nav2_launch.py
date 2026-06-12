import os
import copy
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def set_nested(config, keys, value):
    current = config
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value


def make_robot_nav2_params(base_params, robot):
    params = copy.deepcopy(base_params)

    odom_frame = f'{robot}/odom'
    base_frame = f'{robot}/base_link'
    footprint_frame = f'{robot}/base_footprint'

    set_nested(params, ['amcl', 'ros__parameters', 'base_frame_id'], footprint_frame)
    set_nested(params, ['amcl', 'ros__parameters', 'odom_frame_id'], odom_frame)
    set_nested(params, ['amcl', 'ros__parameters', 'scan_topic'], 'scan')

    set_nested(params, ['bt_navigator', 'ros__parameters', 'robot_base_frame'], base_frame)
    set_nested(params, ['bt_navigator', 'ros__parameters', 'odom_topic'], 'odom')

    set_nested(params, ['controller_server', 'ros__parameters', 'FollowPath', 'BaseObstacle.scale'], 0.08)
    set_nested(
        params,
        ['controller_server', 'ros__parameters', 'goal_checker_plugins'],
        ['general_goal_checker', 'position_goal_checker'])
    set_nested(params, ['controller_server', 'ros__parameters', 'position_goal_checker'], {
        'stateful': False,
        'plugin': 'nav2_controller::SimpleGoalChecker',
        'xy_goal_tolerance': 0.25,
        'yaw_goal_tolerance': 6.2831853,
    })
    set_nested(params, ['controller_server', 'ros__parameters', 'FollowPath', 'xy_goal_tolerance'], 0.25)

    set_nested(params, ['local_costmap', 'local_costmap', 'ros__parameters', 'global_frame'], odom_frame)
    set_nested(params, ['local_costmap', 'local_costmap', 'ros__parameters', 'robot_base_frame'], base_frame)
    set_nested(params, ['local_costmap', 'local_costmap', 'ros__parameters', 'update_frequency'], 10.0)
    set_nested(params, ['local_costmap', 'local_costmap', 'ros__parameters', 'publish_frequency'], 5.0)
    set_nested(params, ['local_costmap', 'local_costmap', 'ros__parameters', 'use_sim_time'], False)
    set_nested(params, ['local_costmap', 'local_costmap', 'ros__parameters', 'inflation_layer', 'inflation_radius'], 0.8)
    set_nested(params, ['local_costmap', 'local_costmap', 'ros__parameters', 'inflation_layer', 'cost_scaling_factor'], 1.5)
    set_nested(params, ['local_costmap', 'local_costmap', 'ros__parameters', 'voxel_layer', 'scan', 'topic'], 'scan')
    set_nested(params, ['local_costmap', 'local_costmap', 'ros__parameters', 'voxel_layer', 'scan', 'observation_persistence'], 0.8)
    set_nested(params, ['local_costmap', 'local_costmap', 'ros__parameters', 'voxel_layer', 'scan', 'obstacle_max_range'], 3.0)
    set_nested(params, ['local_costmap', 'local_costmap', 'ros__parameters', 'voxel_layer', 'scan', 'raytrace_max_range'], 3.5)

    set_nested(params, ['global_costmap', 'global_costmap', 'ros__parameters', 'robot_base_frame'], base_frame)
    set_nested(params, ['global_costmap', 'global_costmap', 'ros__parameters', 'update_frequency'], 2.0)
    set_nested(params, ['global_costmap', 'global_costmap', 'ros__parameters', 'publish_frequency'], 1.0)
    set_nested(params, ['global_costmap', 'global_costmap', 'ros__parameters', 'use_sim_time'], False)
    set_nested(params, ['global_costmap', 'global_costmap', 'ros__parameters', 'inflation_layer', 'inflation_radius'], 0.8)
    set_nested(params, ['global_costmap', 'global_costmap', 'ros__parameters', 'inflation_layer', 'cost_scaling_factor'], 1.5)
    set_nested(params, ['global_costmap', 'global_costmap', 'ros__parameters', 'obstacle_layer', 'scan', 'topic'], 'scan')
    set_nested(params, ['global_costmap', 'global_costmap', 'ros__parameters', 'obstacle_layer', 'scan', 'observation_persistence'], 0.8)
    set_nested(params, ['global_costmap', 'global_costmap', 'ros__parameters', 'obstacle_layer', 'scan', 'obstacle_max_range'], 3.0)
    set_nested(params, ['global_costmap', 'global_costmap', 'ros__parameters', 'obstacle_layer', 'scan', 'raytrace_max_range'], 3.5)

    set_nested(params, ['behavior_server', 'ros__parameters', 'global_frame'], odom_frame)
    set_nested(params, ['behavior_server', 'ros__parameters', 'local_frame'], odom_frame)
    set_nested(params, ['behavior_server', 'ros__parameters', 'robot_base_frame'], base_frame)

    set_nested(params, ['velocity_smoother', 'ros__parameters', 'odom_topic'], 'odom')

    for node_params in params.values():
        ros_params = node_params.get('ros__parameters') if isinstance(node_params, dict) else None
        if ros_params and 'use_sim_time' in ros_params:
            ros_params['use_sim_time'] = False

    return params


def sanitized_ld_library_path():
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


def generate_launch_description():
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    coordinator_dir = get_package_share_directory('multi_turtlebot_coordinator')
    nav2_launch_dir = os.path.join(nav2_bringup_dir, 'launch')
    map_yaml_file = os.path.expanduser('~/empty_room.yaml')
    rviz_config_file = os.path.join(coordinator_dir, 'rviz', 'multi_robot_view.rviz')
    use_rviz = LaunchConfiguration('use_rviz')
    use_central_planner = LaunchConfiguration('use_central_planner')
    rviz_config = LaunchConfiguration('rviz_config')
    
    robots = ['tb_1', 'tb_2', 'tb_3']
    
    # Read the original nav2_params.yaml
    original_params_file = os.path.join(nav2_bringup_dir, 'params', 'nav2_params.yaml')
    with open(original_params_file, 'r') as f:
        original_params = yaml.safe_load(f)
        
    ld = LaunchDescription()
    ld.add_action(SetEnvironmentVariable('LD_LIBRARY_PATH', sanitized_ld_library_path()))
    ld.add_action(DeclareLaunchArgument(
        'use_rviz',
        default_value='True',
        description='Start one RViz window showing all robots'))
    ld.add_action(DeclareLaunchArgument(
        'rviz_config',
        default_value=rviz_config_file,
        description='RViz config used for the combined multi-robot view'))
    ld.add_action(DeclareLaunchArgument(
        'use_central_planner',
        default_value='True',
        description='Start the centralized position-radius fleet planner'))
    
    for robot in robots:
        robot_params = make_robot_nav2_params(original_params, robot)
        
        # Save it to /tmp so the launch file can use it
        tmp_param_file = f'/tmp/nav2_params_{robot}.yaml'
        with open(tmp_param_file, 'w') as f:
            yaml.safe_dump(robot_params, f, sort_keys=False)
            
        bringup_cmd = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(nav2_launch_dir, 'bringup_launch.py')),
            launch_arguments={
                'namespace': robot,
                'use_namespace': 'True',
                'map': map_yaml_file,
                'use_sim_time': 'False',
                'params_file': tmp_param_file,
                'autostart': 'True',
            }.items()
        )
        
        # bringup_launch.py itself pushes the namespace when use_namespace is True,
        # so we don't need PushRosNamespace here.
        ld.add_action(bringup_cmd)

    tf_mux_cmd = Node(
        condition=IfCondition(use_rviz),
        package='multi_turtlebot_coordinator',
        executable='tf_mux.py',
        name='multi_robot_tf_mux',
        parameters=[{'robot_names': robots}],
        output='screen')

    central_planner_cmd = Node(
        condition=IfCondition(use_central_planner),
        package='multi_robot_swarm_planner',
        executable='central_fleet_planner',
        name='central_fleet_planner',
        parameters=[{'robot_names': robots}],
        output='screen')

    rviz_cmd = Node(
        condition=IfCondition(use_rviz),
        package='rviz2',
        executable='rviz2',
        name='multi_robot_rviz',
        arguments=['-d', rviz_config],
        output='screen')

    ld.add_action(tf_mux_cmd)
    ld.add_action(central_planner_cmd)
    ld.add_action(rviz_cmd)
        
    return ld
