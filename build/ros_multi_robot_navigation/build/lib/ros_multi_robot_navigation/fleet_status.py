import argparse
import time

from nav2_msgs.action import NavigateToPose
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from ros_multi_robot_navigation.common import DEFAULT_ROBOTS, namespaced_topic


class FleetStatus(Node):
    def __init__(self, robot_names):
        super().__init__('fleet_status')
        self.robot_names = robot_names
        self.action_clients = {
            robot: ActionClient(self, NavigateToPose, namespaced_topic(robot, 'navigate_to_pose'))
            for robot in robot_names
        }

    def report(self, action_timeout):
        topics = {name for name, _type_names in self.get_topic_names_and_types()}
        rows = []
        for robot in self.robot_names:
            scan = namespaced_topic(robot, 'scan') in topics
            odom = namespaced_topic(robot, 'odom') in topics
            map_topic = namespaced_topic(robot, 'map') in topics
            action_ready = self.action_clients[robot].wait_for_server(timeout_sec=action_timeout)
            rows.append((robot, scan, odom, map_topic, action_ready))
        return rows


def _mark(value):
    return 'yes' if value else 'no'


def main(args=None):
    parser = argparse.ArgumentParser(description='Check whether each robot has basic Nav2 topics/actions.')
    parser.add_argument(
        '--robots',
        default=','.join(DEFAULT_ROBOTS),
        help='Comma-separated robot namespaces.',
    )
    parser.add_argument(
        '--wait',
        type=float,
        default=3.0,
        help='Seconds to wait for ROS discovery before checking topics.',
    )
    parser.add_argument(
        '--action-timeout',
        type=float,
        default=1.0,
        help='Seconds to wait for each NavigateToPose action server.',
    )
    parsed = parser.parse_args(remove_ros_args(args=args)[1:])
    robot_names = [robot.strip().strip('/') for robot in parsed.robots.split(',') if robot.strip()]

    rclpy.init(args=args)
    node = FleetStatus(robot_names)
    try:
        deadline = time.monotonic() + parsed.wait
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        print('robot  scan  odom  map  navigate_to_pose')
        for robot, scan, odom, map_topic, action_ready in node.report(parsed.action_timeout):
            print(
                f'{robot:5}  {_mark(scan):4}  {_mark(odom):4}  '
                f'{_mark(map_topic):3}  {_mark(action_ready)}'
            )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
