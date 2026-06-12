#!/usr/bin/python3

import argparse
import csv
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from tf_transformations import quaternion_from_euler


class MultiRobotInitializer(Node):
    def __init__(self, csv_path, trajectory_id):
        super().__init__('multi_robot_initializer')
        self.csv_path = csv_path
        self.trajectory_id = trajectory_id
        self.initial_poses, self.goal_poses = self.load_poses()
        
        self.initial_pose_pubs = {}
        self.nav_clients = {}
        
        for robot_id in self.initial_poses.keys():
            ns = f"tb_{robot_id + 1}"
            pub = self.create_publisher(PoseWithCovarianceStamped, f'/{ns}/initialpose', 10)
            self.initial_pose_pubs[robot_id] = pub
            
            client = ActionClient(self, NavigateToPose, f'/{ns}/navigate_to_pose')
            self.nav_clients[robot_id] = client
            
    def load_poses(self):
        with open(self.csv_path, newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            rows = [row for row in reader if int(row["trajectory_id"]) == self.trajectory_id]

        if not rows:
            raise ValueError(f"No rows found for trajectory_id={self.trajectory_id}")

        if {"start_x_m", "start_y_m", "start_theta_rad", "goal_x_m", "goal_y_m", "goal_theta_rad"} <= fieldnames:
            return self.load_mission_poses(rows)

        if {"step", "x_m", "y_m", "theta_rad"} <= fieldnames:
            return self.load_trajectory_poses(rows, fieldnames)

        raise ValueError(
            "CSV must contain either mission columns "
            "(start_x_m/start_y_m/start_theta_rad/goal_x_m/goal_y_m/goal_theta_rad) "
            "or trajectory columns (step/x_m/y_m/theta_rad).")

    def load_mission_poses(self, rows):
        initial_poses = {}
        goal_poses = {}

        for row in rows:
            robot_id = int(row["robot_id"])
            initial_poses[robot_id] = {
                "x": float(row["start_x_m"]),
                "y": float(row["start_y_m"]),
                "theta": float(row["start_theta_rad"])
            }
            goal_poses[robot_id] = {
                "x": float(row["goal_x_m"]),
                "y": float(row["goal_y_m"]),
                "theta": float(row["goal_theta_rad"])
            }

        return initial_poses, goal_poses

    def load_trajectory_poses(self, rows, fieldnames):
        by_robot = {}
        for row in rows:
            by_robot.setdefault(int(row["robot_id"]), []).append(row)

        initial_poses = {}
        goal_poses = {}

        for robot_id, robot_rows in by_robot.items():
            robot_rows.sort(key=lambda row: int(row["step"]))
            first = robot_rows[0]
            last = robot_rows[-1]

            initial_poses[robot_id] = {
                "x": float(first["x_m"]),
                "y": float(first["y_m"]),
                "theta": float(first["theta_rad"])
            }

            if {"goal_x_m", "goal_y_m"} <= fieldnames:
                goal_poses[robot_id] = {
                    "x": float(last["goal_x_m"]),
                    "y": float(last["goal_y_m"]),
                    "theta": float(last["theta_rad"])
                }
            else:
                goal_poses[robot_id] = {
                    "x": float(last["x_m"]),
                    "y": float(last["y_m"]),
                    "theta": float(last["theta_rad"])
                }

        return initial_poses, goal_poses

    def send_initial_poses(self):
        self.get_logger().info("Sending initial poses to AMCL...")
        for robot_id, pose in self.initial_poses.items():
            msg = PoseWithCovarianceStamped()
            msg.header.frame_id = 'map'
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.pose.position.x = pose['x']
            msg.pose.pose.position.y = pose['y']
            
            q = quaternion_from_euler(0, 0, pose['theta'])
            msg.pose.pose.orientation.x = q[0]
            msg.pose.pose.orientation.y = q[1]
            msg.pose.pose.orientation.z = q[2]
            msg.pose.pose.orientation.w = q[3]
            
            # small covariance
            msg.pose.covariance[0] = 0.05
            msg.pose.covariance[7] = 0.05
            msg.pose.covariance[35] = 0.05
            
            # publish a few times to ensure it is received
            for _ in range(5):
                self.initial_pose_pubs[robot_id].publish(msg)
            self.get_logger().info(
                f"tb_{robot_id + 1} initial pose sent: ({pose['x']:.2f}, {pose['y']:.2f})")

    def send_navigation_goals(self):
        self.get_logger().info("Sending NavigateToPose goals...")
        for robot_id, pose in self.goal_poses.items():
            client = self.nav_clients[robot_id]
            if not client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error(f"NavigateToPose action server not available for tb_{robot_id + 1}")
                continue
                
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose.header.frame_id = 'map'
            goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
            goal_msg.pose.pose.position.x = pose['x']
            goal_msg.pose.pose.position.y = pose['y']
            
            q = quaternion_from_euler(0, 0, pose['theta'])
            goal_msg.pose.pose.orientation.x = q[0]
            goal_msg.pose.pose.orientation.y = q[1]
            goal_msg.pose.pose.orientation.z = q[2]
            goal_msg.pose.pose.orientation.w = q[3]
            
            client.send_goal_async(goal_msg)
            self.get_logger().info(
                f"tb_{robot_id + 1} navigating to goal pose ({pose['x']:.2f}, {pose['y']:.2f})...")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--trajectory-id", type=int, default=0)
    args = parser.parse_args()

    rclpy.init()
    node = MultiRobotInitializer(args.csv_path, args.trajectory_id)
    
    # Wait a bit for publishers to connect
    import time
    time.sleep(2.0)
    
    node.send_initial_poses()
    time.sleep(1.0)
    node.send_navigation_goals()
    
    # We could wait for results here, but spinning for a bit is fine
    # Users can observe RViz.
    end_time = time.time() + 5.0
    while rclpy.ok() and time.time() < end_time:
        rclpy.spin_once(node, timeout_sec=0.1)
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
