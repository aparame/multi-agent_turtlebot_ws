import argparse
import base64
import os
import shlex
import signal
import socket
import subprocess
import sys

from ament_index_python.packages import get_package_share_directory
import yaml


def _load_config(path):
    with open(path, 'r', encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def _remote_command(domain_id, namespace, model, lds_model, usb_port):
    commands = [
        'set -e',
        'source /opt/ros/humble/setup.bash',
        'if [ -f ~/turtlebot3_ws/install/setup.bash ]; then source ~/turtlebot3_ws/install/setup.bash; fi',
        'pkill -f "ros2 launch turtlebot3_bringup [r]obot.launch.py" || true',
        'pkill -f "[h]lds_laser_publisher|[l]d08|[s]ingle_lidar_node|[r]obot_state_publisher|[t]urtlebot3_ros" || true',
        'sleep 2',
        f'export ROS_DOMAIN_ID={domain_id}',
        f'export TURTLEBOT3_MODEL={shlex.quote(model)}',
        f'export LDS_MODEL={shlex.quote(lds_model)}',
        (
            'exec ros2 launch turtlebot3_bringup robot.launch.py '
            f'namespace:={shlex.quote(namespace)} '
            f'usb_port:={shlex.quote(usb_port)} '
            'use_sim_time:=false'
        ),
    ]
    script = '\n'.join(commands)
    encoded_script = base64.b64encode(script.encode('utf-8')).decode('ascii')
    return f'printf %s {shlex.quote(encoded_script)} | base64 -d | bash'


def _ssh_command(robot, domain_id, model, lds_model, usb_port, identity_file=None):
    destination = f"{robot['user']}@{robot['host']}"
    namespace = robot.get('namespace', robot['name'])
    command = [
        'ssh',
        '-tt',
        '-o',
        'ServerAliveInterval=10',
        '-o',
        'ServerAliveCountMax=3',
    ]
    if identity_file:
        command.extend(['-i', os.path.expanduser(identity_file)])
    command.extend([
        destination,
        _remote_command(domain_id, namespace, model, lds_model, usb_port),
    ])
    return command


def _check_ssh_port(robot, timeout):
    try:
        with socket.create_connection((robot['host'], 22), timeout=timeout):
            return True, ''
    except OSError as exc:
        return False, str(exc)


def main(argv=None):
    try:
        default_config = os.path.join(
            get_package_share_directory('ros_multi_robot_navigation'),
            'config',
            'robots.yaml',
        )
    except Exception:
        default_config = os.path.join(os.getcwd(), 'ROS_multi_robot_navigation', 'config', 'robots.yaml')

    parser = argparse.ArgumentParser(
        description='SSH into the three lab TurtleBot3 Burgers and start namespaced robot bringup.'
    )
    parser.add_argument(
        '--config',
        default=default_config,
        help='Path to robots.yaml. Use the installed share path or this source-tree path.',
    )
    parser.add_argument(
        '--robot',
        action='append',
        dest='robots',
        help='Only start this robot name. May be passed more than once.',
    )
    parser.add_argument(
        '--domain-id',
        type=int,
        default=None,
        help='Shared ROS_DOMAIN_ID for multi-robot mode. Defaults to shared_domain_id from robots.yaml.',
    )
    parser.add_argument('--usb-port', default='/dev/ttyACM0', help='OpenCR USB port on each robot.')
    parser.add_argument('--dry-run', action='store_true', help='Print SSH commands without running them.')
    parser.add_argument(
        '--identity-file',
        help='SSH private key to use, for example ~/.ssh/turtlebot_lab_ed25519.',
    )
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Start robots one at a time. Since robot bringup keeps running, this is mainly for debugging one selected robot.',
    )
    parser.add_argument(
        '--skip-reachability-check',
        action='store_true',
        help='Skip the TCP port 22 preflight check before starting SSH sessions.',
    )
    parser.add_argument(
        '--ssh-timeout',
        type=float,
        default=3.0,
        help='Seconds to wait while checking whether each robot SSH port is reachable.',
    )
    args = parser.parse_args(argv)

    if not os.path.exists(args.config):
        print(f'Config file not found: {args.config}', file=sys.stderr)
        print('From an installed workspace, pass --config $(ros2 pkg prefix ros_multi_robot_navigation)/share/ros_multi_robot_navigation/config/robots.yaml', file=sys.stderr)
        return 2

    config = _load_config(args.config)
    selected = set(args.robots or [])
    robots = [
        robot for robot in config['robots']
        if not selected or robot['name'] in selected or robot.get('namespace') in selected
    ]

    if not robots:
        print('No robots selected.', file=sys.stderr)
        return 2

    domain_id = args.domain_id if args.domain_id is not None else config.get('shared_domain_id', 10)
    model = config.get('model', 'burger')
    lds_model = config.get('lds_model', 'LDS-01')

    if not args.dry_run and not args.skip_reachability_check:
        unreachable = []
        for robot in robots:
            ok, reason = _check_ssh_port(robot, args.ssh_timeout)
            if not ok:
                unreachable.append((robot, reason))

        if unreachable:
            print('Cannot reach one or more TurtleBots on SSH port 22:', file=sys.stderr)
            for robot, reason in unreachable:
                print(
                    f"  - {robot['name']} ({robot['user']}@{robot['host']}): {reason}",
                    file=sys.stderr,
                )
            print('', file=sys.stderr)
            print('Check that:', file=sys.stderr)
            print('  1. The PC is connected to the same lab WiFi as the robots, e.g. NETGEAR11-5G.', file=sys.stderr)
            print('  2. The robots are powered on and have joined that network.', file=sys.stderr)
            print('  3. The IP addresses in config/robots.yaml still match the robots.', file=sys.stderr)
            print('  4. You can SSH manually into each robot before starting multi-robot bringup.', file=sys.stderr)
            print('', file=sys.stderr)
            print('Manual checks:', file=sys.stderr)
            for robot, _reason in unreachable:
                print(f"  ssh {robot['user']}@{robot['host']}", file=sys.stderr)
            return 1

    processes = []
    for robot in robots:
        command = _ssh_command(
            robot,
            domain_id,
            model,
            lds_model,
            args.usb_port,
            identity_file=args.identity_file,
        )
        printable = ' '.join(shlex.quote(part) for part in command)
        print(f"[{robot['name']}] {printable}")
        if not args.dry_run:
            if args.sequential:
                result = subprocess.run(command)
                if result.returncode != 0:
                    return result.returncode
            else:
                processes.append(subprocess.Popen(command))

    if args.dry_run or args.sequential:
        return 0

    try:
        return max(process.wait() for process in processes)
    except KeyboardInterrupt:
        for process in processes:
            process.send_signal(signal.SIGINT)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
