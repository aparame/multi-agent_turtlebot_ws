import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage

from ros_multi_robot_navigation.common import DEFAULT_ROBOTS, namespaced_topic


class FleetRvizBridge(Node):
    def __init__(self):
        super().__init__('fleet_rviz_bridge')
        self.declare_parameter('robot_names', list(DEFAULT_ROBOTS))
        robot_names = self.get_parameter('robot_names').value
        self.robot_names = [name.strip().strip('/') for name in robot_names if name.strip()]

        tf_qos = QoSProfile(depth=100)
        static_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.tf_pub = self.create_publisher(TFMessage, '/tf', tf_qos)
        self.tf_static_pub = self.create_publisher(TFMessage, '/tf_static', static_qos)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', map_qos)

        self._bridge_subscriptions = []
        for robot in self.robot_names:
            self._bridge_subscriptions.append(
                self.create_subscription(
                    TFMessage,
                    namespaced_topic(robot, 'tf'),
                    self.tf_pub.publish,
                    tf_qos,
                )
            )
            self._bridge_subscriptions.append(
                self.create_subscription(
                    TFMessage,
                    namespaced_topic(robot, 'tf_static'),
                    self.tf_static_pub.publish,
                    static_qos,
                )
            )

        if self.robot_names:
            first_robot_map = namespaced_topic(self.robot_names[0], 'map')
            self._bridge_subscriptions.append(
                self.create_subscription(OccupancyGrid, first_robot_map, self.map_pub.publish, map_qos)
            )
            self.get_logger().info(
                f'Bridging TF for {", ".join(self.robot_names)} and map from {first_robot_map} to /map'
            )


def main(args=None):
    rclpy.init(args=args)
    node = FleetRvizBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
