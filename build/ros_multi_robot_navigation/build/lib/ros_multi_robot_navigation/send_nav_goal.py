import argparse

from nav2_msgs.action import NavigateToPose
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from ros_multi_robot_navigation.common import namespaced_topic, yaw_to_quaternion


class GoalSender(Node):
    def __init__(self, robot_name):
        super().__init__('send_nav_goal')
        self.robot_name = robot_name
        self.client = ActionClient(
            self,
            NavigateToPose,
            namespaced_topic(robot_name, 'navigate_to_pose'),
        )

    def send(self, x, y, yaw, timeout):
        if not self.client.wait_for_server(timeout_sec=timeout):
            raise RuntimeError(f'NavigateToPose action server is not available for {self.robot_name}')

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation = yaw_to_quaternion(yaw)

        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if not handle.accepted:
            raise RuntimeError(f'Goal rejected by {self.robot_name}')

        self.get_logger().info(f'Goal accepted by {self.robot_name}; waiting for result')
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return result_future.result()


def main(args=None):
    parser = argparse.ArgumentParser(description='Send a NavigateToPose goal to one robot.')
    parser.add_argument('robot', help='Robot namespace, for example tb3_1.')
    parser.add_argument('x', type=float, help='Goal x in the map frame.')
    parser.add_argument('y', type=float, help='Goal y in the map frame.')
    parser.add_argument('yaw', type=float, help='Goal yaw in radians.')
    parser.add_argument('--server-timeout', type=float, default=10.0)
    parsed = parser.parse_args(remove_ros_args(args=args)[1:])

    rclpy.init(args=args)
    node = GoalSender(parsed.robot)
    try:
        result = node.send(parsed.x, parsed.y, parsed.yaw, parsed.server_timeout)
        node.get_logger().info(f'{parsed.robot} navigation finished with status {result.status}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
