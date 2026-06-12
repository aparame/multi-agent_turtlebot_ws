#!/usr/bin/python3

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from tf2_msgs.msg import TFMessage


class TfMux(Node):
    def __init__(self):
        super().__init__('multi_robot_tf_mux')
        self.declare_parameter('robot_names', ['tb_1', 'tb_2', 'tb_3'])
        robot_names = list(self.get_parameter('robot_names').value)

        dynamic_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        static_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.tf_pub = self.create_publisher(TFMessage, '/tf', dynamic_qos)
        self.tf_static_pub = self.create_publisher(TFMessage, '/tf_static', static_qos)
        self.tf_static_cache = {}

        for robot_name in robot_names:
            self.create_subscription(
                TFMessage,
                f'/{robot_name}/tf',
                self.publish_tf,
                dynamic_qos,
            )
            self.create_subscription(
                TFMessage,
                f'/{robot_name}/tf_static',
                self.publish_static_tf,
                static_qos,
            )

        self.get_logger().info(
            f"Relaying namespaced TF for RViz: {', '.join(robot_names)}")

    def publish_tf(self, msg):
        if msg.transforms:
            self.tf_pub.publish(msg)

    def publish_static_tf(self, msg):
        changed = False
        for transform in msg.transforms:
            key = (transform.header.frame_id, transform.child_frame_id)
            self.tf_static_cache[key] = transform
            changed = True

        if changed:
            self.tf_static_pub.publish(
                TFMessage(transforms=list(self.tf_static_cache.values())))


def main():
    rclpy.init()
    node = TfMux()
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
