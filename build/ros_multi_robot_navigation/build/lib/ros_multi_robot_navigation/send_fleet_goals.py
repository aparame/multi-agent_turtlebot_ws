import argparse

from nav2_msgs.action import NavigateToPose
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from ros_multi_robot_navigation.common import namespaced_topic, yaw_to_quaternion


class FleetGoalSender(Node):
    def __init__(self, goals):
        super().__init__('send_fleet_goals')
        self.goals = goals
        self.clients = {
            robot: ActionClient(self, NavigateToPose, namespaced_topic(robot, 'navigate_to_pose'))
            for robot, _x, _y, _yaw in goals
        }

    def send_all(self, timeout):
        goal_futures = {}
        for robot, x, y, yaw in self.goals:
            client = self.clients[robot]
            if not client.wait_for_server(timeout_sec=timeout):
                raise RuntimeError(f'NavigateToPose action server is not available for {robot}')

            goal = NavigateToPose.Goal()
            goal.pose.header.frame_id = 'map'
            goal.pose.header.stamp = self.get_clock().now().to_msg()
            goal.pose.pose.position.x = x
            goal.pose.pose.position.y = y
            goal.pose.pose.orientation = yaw_to_quaternion(yaw)

            self.get_logger().info(f'Sending {robot} goal: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}')
            goal_futures[robot] = client.send_goal_async(goal)

        goal_handles = {}
        for robot, future in goal_futures.items():
            rclpy.spin_until_future_complete(self, future)
            handle = future.result()
            if handle is None or not handle.accepted:
                raise RuntimeError(f'Goal rejected by {robot}')
            goal_handles[robot] = handle
            self.get_logger().info(f'{robot} accepted goal')

        result_futures = {
            robot: handle.get_result_async()
            for robot, handle in goal_handles.items()
        }

        while result_futures:
            rclpy.spin_once(self, timeout_sec=0.1)
            finished = [
                robot for robot, future in result_futures.items()
                if future.done()
            ]
            for robot in finished:
                result = result_futures.pop(robot).result()
                self.get_logger().info(f'{robot} finished with status {result.status}')


def _parse_goal(values):
    robot, x, y, yaw = values
    return robot.strip().strip('/'), float(x), float(y), float(yaw)


def main(args=None):
    parser = argparse.ArgumentParser(description='Send NavigateToPose goals to multiple robots.')
    parser.add_argument(
        '--goal',
        action='append',
        nargs=4,
        metavar=('ROBOT', 'X', 'Y', 'YAW'),
        required=True,
        help='Robot namespace and map-frame goal. Example: --goal tb3_1 2.0 0.5 0.0',
    )
    parser.add_argument('--server-timeout', type=float, default=10.0)
    parsed = parser.parse_args(remove_ros_args(args=args)[1:])

    goals = [_parse_goal(goal) for goal in parsed.goal]

    rclpy.init(args=args)
    node = FleetGoalSender(goals)
    try:
        node.send_all(parsed.server_timeout)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
