import argparse
import time

from geometry_msgs.msg import PoseWithCovarianceStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.utilities import remove_ros_args

from ros_multi_robot_navigation.common import namespaced_topic, yaw_to_quaternion


class InitialPosePublisher(Node):
    def __init__(self, robot_name):
        super().__init__('set_initial_pose')
        self.publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            namespaced_topic(robot_name, 'initialpose'),
            QoSProfile(depth=10),
        )


def _build_pose(x, y, yaw):
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = 'map'
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.orientation = yaw_to_quaternion(yaw)
    msg.pose.covariance[0] = 0.25
    msg.pose.covariance[7] = 0.25
    msg.pose.covariance[35] = 0.0685
    return msg


def main(args=None):
    parser = argparse.ArgumentParser(description='Publish an AMCL initial pose for one robot.')
    parser.add_argument('robot', help='Robot namespace, for example tb3_1.')
    parser.add_argument('x', type=float, help='Initial x in the map frame.')
    parser.add_argument('y', type=float, help='Initial y in the map frame.')
    parser.add_argument('yaw', type=float, help='Initial yaw in radians.')
    parser.add_argument('--repeat', type=int, default=5, help='Number of pose messages to publish.')
    parser.add_argument('--period', type=float, default=0.2, help='Seconds between repeated messages.')
    parsed = parser.parse_args(remove_ros_args(args=args)[1:])

    rclpy.init(args=args)
    node = InitialPosePublisher(parsed.robot)
    try:
        pose = _build_pose(parsed.x, parsed.y, parsed.yaw)
        for _ in range(parsed.repeat):
            pose.header.stamp = node.get_clock().now().to_msg()
            node.publisher.publish(pose)
            rclpy.spin_once(node, timeout_sec=0.05)
            time.sleep(parsed.period)
        node.get_logger().info(
            f'Published initial pose for {parsed.robot}: x={parsed.x:.2f}, y={parsed.y:.2f}, yaw={parsed.yaw:.2f}'
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
