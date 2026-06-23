# 🐢 Multi-Agent TurtleBot3 CV Navigation Stack

[![ROS 2 Humble](https://img.shields.io/badge/ROS2-Humble-blue?logo=ros&logoColor=white)](https://docs.ros.org/en/humble/)
[![Python](https://img.shields.io/badge/Python-3.10-green?logo=python&logoColor=white)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C++-17-orange?logo=c%2B%2B&logoColor=white)](https://isocpp.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red?logo=opencv&logoColor=white)](https://opencv.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-green?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

This repository contains a high-performance, lightweight overhead-camera localization and direct velocity-control stack for **multi-agent control of three TurtleBot3 Burger robots**. 

The system operates centrally without the overhead of AMCL, Nav2, map servers, planner/controller servers, or lidar-based navigation.

> [!NOTE]  
> This package requires the standard TurtleBot3 ROS 2 dependencies. For hardware configuration and environment setup, please refer to the official [ROBOTIS TurtleBot3 e-Manual / ROS 2 Setup Guide](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/).

---

## ✨ Features

* **Real-time Overhead Localization**: Fuses overhead camera coordinates with local odometry yaw deltas using an Extended Kalman Filter (EKF).
* **Scheduled Planner Mode**: Generates conflict-free joint schedules offline using Conflict-Based Search (CBS) or Prioritized Planning, tracked by a path follower and priority-aware ORCA local velocity filter.
* **Interactive Operator GUI**: Facilitates easy point-and-click robot localization, initial heading setup, target assignment, and real-time path visualization.

---

## ⚙️ Quick Start

### 1. Clone & Build the Workspace
Clone this repository and compile the workspace using `colcon`:
```bash
git clone <repository_url>
cd ~/turtlebot_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### 2. Connect to Wi-Fi
> [!IMPORTANT]
> **Network Connection**: Connect the operator PC to the dedicated TurtleBot3 Wi-Fi network (SSID: **`netgear11`**). Do **not** connect via wired Ethernet, as the robots communicate exclusively over this wireless network.

### 3. Remote Robot Bringup (Terminal A)
Ensure the robots and your PC share the same domain and start remote hardware bringup over SSH:
```bash
cd ~/turtlebot_ws
source install/setup.bash
export ROS_DOMAIN_ID=30
ros2 run ros_multi_robot_navigation bringup_lab_robots \
  --identity-file ~/.ssh/turtlebot_lab_ed25519
```

### 4. Run the Fleet Controller (Terminal B)
Start the central operator control node:
```bash
cd ~/turtlebot_ws
source install/setup.bash
export ROS_DOMAIN_ID=30
ros2 launch cv_localization cv_mppi_direct.launch.py
```

---

## 🎮 Operator GUI Workflow

Once the **"CV Scheduled Multi-Agent Control"** GUI window is visible:

1. **Identify**: Click on each visible robot blob (`tb_1` 🔴, `tb_2` 🔵, `tb_3` 🟢) to assign identities.
2. **Heading**: Click a second point slightly ahead of each robot's front to initialize the EKF yaw.
3. **Goals**: Click goal coordinates for each robot in sequential order (`tb_1` → `tb_2` → `tb_3`).
4. **Run**: Press **`Enter`** to start motion.

### Keyboard Shortcuts
| Key | Action |
| :---: | :--- |
| **`Enter`** | Commences motion control after goal setting. |
| **`Space`** | **Emergency Stop** (commands zero velocity to all robots instantly). |
| **`Esc`** | **Emergency Stop** (commands zero velocity to all robots instantly). |
| **`r`** | Resets and clears goals and planners. |
| **`q`** | Stops all robots and terminates the GUI. |

---

## 📋 Robot Fleet Registry

Configuration file location: `src/multi_robot_navigation_ROS2/config/robots.yaml`

| Robot | Host IP | SSH Login | ROS Domain ID |
| :---: | :--- | :--- | :---: |
| **`tb_1`** | `192.168.1.20` | `turtlebot@192.168.1.20` | `30` |
| **`tb_2`** | `192.168.1.15` | `ubuntu@192.168.1.15` | `30` |
| **`tb_3`** | `192.168.1.16` | `ubuntu@192.168.1.16` | `30` |

---

## 📡 ROS Topics & Services

### Subscribed & Published Topics
* `/tb_N/cv_pose` (Pub): Fused camera/EKF robot pose from GUI.
* `/tb_N/odom` (Sub): Local odometry yaw deltas for fusion EKF.
* `/tb_N/mppi_goal` (Pub): Position target coordinates.
* `/tb_N/offline_plan` (Pub): Computed multi-agent scheduled path preview.
* `/tb_N/cmd_vel` (Pub): Final velocity command dispatched to the robot.
* `/fleet_mppi/status` (Pub): Global fleet controller event logs.

### Operational Services
* `/fleet_mppi/start` (`Trigger`): Begins path execution.
* `/fleet_mppi/stop` (`Trigger`): Immediately halts all robots.
* `/fleet_mppi/clear_goals` (`Trigger`): Resets goals.
* `/fleet_mppi/plan` (`Trigger`): Manually triggers offline path generation.

---

## 📚 Advanced References & Configuration

All system architecture, package structure details, configuration matrices, and setup tools are located in **[DETAILS.md](DETAILS.md)**:

* 📐 **[System Architecture](DETAILS.md#1-system-architecture)**: Signal flow, camera tracking pipeline, and safety loop diagram.
* 📁 **[Workspace Layout](DETAILS.md#2-workspace-layout)**: Detailed outline of repository directories and internal package boundaries.
* 🧠 **[Algorithm Planners](DETAILS.md#3-algorithm-deep-dive)**: Deep dive into CBS, SIPP, and ORCA.
* 🔑 **[SSH Passwordless Configuration](DETAILS.md#4-ssh-setup-for-robot-bringup)**: Setting up secure, automated remote bringup.
* 📷 **[Workspace & Homography Calibration](DETAILS.md#5-workspace--camera-calibration)**: How to align camera pixels to physical world coordinates.
* 🖼️ **[Static Background Capture](DETAILS.md#6-one-time-background-capture)**: Setting up static background subtraction.
* ⚙️ **[Planner & Control Parameters Reference](DETAILS.md#7-planning-control-parameters-reference)**: Descriptions of variables in `config.yaml`.
* 🛡️ **[Safety Envelope Guards](DETAILS.md#8-safety-behavior)**: Trigger bounds and automatic shutdown triggers.
* 🩺 **[Troubleshooting & Diagnostics](DETAILS.md#9-troubleshooting-guide)**: Common errors, detection adjustments, and networking resolutions.
