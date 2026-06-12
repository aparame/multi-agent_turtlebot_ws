import argparse
import os
import shlex
import subprocess
import sys

from ament_index_python.packages import get_package_share_directory
import yaml


def _load_config(path):
    with open(path, 'r', encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def _remote_script(domain_id, model, lds_model, namespace):
    commands = [
        'set +e',
        'source /opt/ros/humble/setup.bash',
        'if [ -f ~/turtlebot3_ws/install/setup.bash ]; then source ~/turtlebot3_ws/install/setup.bash; fi',
        f'export ROS_DOMAIN_ID={domain_id}',
        f'export TURTLEBOT3_MODEL={shlex.quote(model)}',
        f'export LDS_MODEL={shlex.quote(lds_model)}',
        'echo "=== host ==="',
        'hostname',
        'echo "=== ros env ==="',
        'echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"',
        'echo "ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-unset}"',
        'echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-unset}"',
        'echo "TURTLEBOT3_MODEL=$TURTLEBOT3_MODEL"',
        'echo "LDS_MODEL=$LDS_MODEL"',
        'echo "=== devices ==="',
        'ls -l /dev/ttyACM* /dev/ttyUSB* 2>&1',
        'echo "=== turtlebot packages ==="',
        'ros2 pkg prefix turtlebot3_bringup 2>&1',
        'ros2 pkg prefix turtlebot3_node 2>&1',
        'echo "=== robot bringup processes ==="',
        'pgrep -af "turtlebot3|hlds|ld08|coin_d4|robot_state_publisher|ros2 launch" 2>&1',
        f'echo "=== local topics containing /{namespace} ==="',
        f'ros2 topic list 2>/dev/null | grep "/{namespace}" || true',
        f'echo "=== expected topic types for /{namespace}/scan and /{namespace}/odom ==="',
        f'ros2 topic type /{namespace}/scan 2>&1',
        f'ros2 topic type /{namespace}/odom 2>&1',
    ]
    return 'bash -lc ' + shlex.quote(' && '.join(commands))


def _ssh_command(robot, script, identity_file=None):
    command = [
        'ssh',
        '-tt',
        '-o',
        'ConnectTimeout=5',
        '-o',
        'ServerAliveInterval=10',
        '-o',
        'ServerAliveCountMax=3',
    ]
    if identity_file:
        command.extend(['-i', os.path.expanduser(identity_file)])
    command.extend([f"{robot['user']}@{robot['host']}", script])
    return command


def main(argv=None):
    try:
        default_config = os.path.join(
            get_package_share_directory('ros_multi_robot_navigation'),
            'config',
            'robots.yaml',
        )
    except Exception:
        default_config = os.path.join(os.getcwd(), 'config', 'robots.yaml')

    parser = argparse.ArgumentParser(description='Run SSH-based TurtleBot bringup diagnostics.')
    parser.add_argument('--config', default=default_config)
    parser.add_argument('--robot', action='append', dest='robots')
    parser.add_argument('--identity-file', help='SSH private key, for example ~/.ssh/turtlebot_lab_ed25519.')
    parser.add_argument('--domain-id', type=int)
    args = parser.parse_args(argv)

    if not os.path.exists(args.config):
        print(f'Config file not found: {args.config}', file=sys.stderr)
        return 2

    config = _load_config(args.config)
    selected = set(args.robots or [])
    robots = [
        robot for robot in config['robots']
        if not selected or robot['name'] in selected or robot.get('namespace') in selected
    ]
    domain_id = args.domain_id if args.domain_id is not None else config.get('shared_domain_id', 10)
    model = config.get('model', 'burger')
    lds_model = config.get('lds_model', 'LDS-01')

    status = 0
    for robot in robots:
        namespace = robot.get('namespace', robot['name'])
        print(f"\n######## {robot['name']} ({robot['user']}@{robot['host']}) ########")
        script = _remote_script(domain_id, model, lds_model, namespace)
        result = subprocess.run(_ssh_command(robot, script, args.identity_file))
        status = max(status, result.returncode)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
