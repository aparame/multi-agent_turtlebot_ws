#!/usr/bin/python3

from functools import partial

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


STATUS_NAMES = {
    GoalStatus.STATUS_UNKNOWN: 'unknown',
    GoalStatus.STATUS_ACCEPTED: 'accepted',
    GoalStatus.STATUS_EXECUTING: 'executing',
    GoalStatus.STATUS_CANCELING: 'canceling',
    GoalStatus.STATUS_SUCCEEDED: 'succeeded',
    GoalStatus.STATUS_CANCELED: 'canceled',
    GoalStatus.STATUS_ABORTED: 'aborted',
}


class GoalPoseRouter(Node):
    def __init__(self):
        super().__init__('multi_robot_goal_pose_router')
        self.declare_parameter('robot_names', ['tb_1', 'tb_2', 'tb_3'])
        self.declare_parameter('goal_frame', 'map')

        self.robot_names = list(self.get_parameter('robot_names').value)
        self.goal_frame = self.get_parameter('goal_frame').value
        self.action_clients = {}
        self.goal_counters = {robot_name: 0 for robot_name in self.robot_names}

        for robot_name in self.robot_names:
            self.action_clients[robot_name] = ActionClient(
                self,
                NavigateToPose,
                f'/{robot_name}/navigate_to_pose',
            )
            self.create_subscription(
                PoseStamped,
                f'/{robot_name}/goal_pose',
                partial(self.goal_pose_callback, robot_name),
                10,
            )

        routed_topics = ', '.join(
            f'/{robot_name}/goal_pose -> /{robot_name}/navigate_to_pose'
            for robot_name in self.robot_names
        )
        self.get_logger().info(f'Routing RViz goals: {routed_topics}')

    def goal_pose_callback(self, robot_name, pose_msg):
        action_client = self.action_clients[robot_name]

        if not action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warning(
                f'/{robot_name}/navigate_to_pose is not available yet; '
                f'ignored goal from /{robot_name}/goal_pose'
            )
            return

        goal_pose = PoseStamped()
        goal_pose.header = pose_msg.header
        goal_pose.pose = pose_msg.pose
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        if not goal_pose.header.frame_id:
            goal_pose.header.frame_id = self.goal_frame

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose
        goal_msg.behavior_tree = ''

        self.goal_counters[robot_name] += 1
        goal_id = self.goal_counters[robot_name]

        self.get_logger().info(
            f'Sending {robot_name} goal #{goal_id}: '
            f'x={goal_pose.pose.position.x:.3f}, '
            f'y={goal_pose.pose.position.y:.3f}, '
            f'frame={goal_pose.header.frame_id}'
        )

        future = action_client.send_goal_async(goal_msg)
        future.add_done_callback(
            partial(self.goal_response_callback, robot_name, goal_id)
        )

    def goal_response_callback(self, robot_name, goal_id, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(
                f'{robot_name} goal #{goal_id} failed while sending: {exc}'
            )
            return

        if not goal_handle.accepted:
            self.get_logger().warning(f'{robot_name} goal #{goal_id} was rejected')
            return

        self.get_logger().info(
            f'{robot_name} goal #{goal_id} accepted; Nav2 is planning/executing'
        )
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            partial(self.result_callback, robot_name, goal_id)
        )

    def result_callback(self, robot_name, goal_id, future):
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error(
                f'{robot_name} goal #{goal_id} failed while waiting for result: {exc}'
            )
            return

        status_name = STATUS_NAMES.get(result.status, str(result.status))
        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'{robot_name} goal #{goal_id} succeeded')
        else:
            self.get_logger().warning(
                f'{robot_name} goal #{goal_id} finished with status: {status_name}'
            )


def main():
    rclpy.init()
    node = GoalPoseRouter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
