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

## 🎬 Multi-Agent Control in Action

Here is the TurtleBot3 Burger fleet navigating conflicts and reaching target goals using our localization and control stack:

### Episode 1 (0:02 - 0:28)
![Episode 1](videos/episode_1.gif)

### Episode 2 (0:39 - 1:26)
![Episode 2](videos/episode_2.gif)

### Episode 3 (1:35 - 2:09)
![Episode 3](videos/episode_3.gif)


---

## ✨ Features

* **Real-time Overhead Localization**: Fuses overhead camera coordinates with local odometry yaw deltas using an Extended Kalman Filter (EKF).
* **Scheduled Planner Mode**: Generates conflict-free joint schedules offline using Conflict-Based Search (CBS) or Prioritized Planning, tracked by a path follower and priority-aware ORCA local velocity filter.
* **RL Checkpoint Mode**: Evaluates an AERO-MARL graph-based Transformer policy directly on physical hardware, wrapped in robust runtime safety filters.
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
Start the central operator control node using one of the two modes:

#### Option A: Model-Based (Scheduled CBS & ORCA Planner)
```bash
cd ~/turtlebot_ws
source install/setup.bash
export ROS_DOMAIN_ID=30
ros2 launch cv_localization cv_mppi_direct.launch.py
```

#### Option B: RL-Based (AERO-MARL Checkpoint Policy)
To evaluate the MARL policy, run the launch file pointing to the repository's saved model checkpoint:
```bash
cd ~/turtlebot_ws
source install/setup.bash
export ROS_DOMAIN_ID=30
ros2 launch cv_localization cv_rl_direct.launch.py \
  model_dir:=/home/adi2440/turtlebot_ws/models/transformer_800.pt \
  aero_marl_root:=/home/adi2440/Desktop/MARL_Shahil_Aditya/AERO-MARL
```

#### Option C: RL-Based Live MA-VLCM Data Collection
To collect TurtleBot lab episodes for MA-VLCM fine tuning, launch the RL controller,
GUI, and WebDataset collector together:
```bash
cd ~/turtlebot_ws
source install/setup.bash
export ROS_DOMAIN_ID=30
ros2 launch cv_localization cv_rl_vlcm_collect.launch.py \
  model_dir:=/home/adi2440/turtlebot_ws/models/transformer_800.pt \
  aero_marl_root:=/home/adi2440/Desktop/MARL_Shahil_Aditya/AERO-MARL \
  episodes:=100 \
  dataset_dir:=/home/adi2440/Desktop/MARL_Shahil_Aditya/MA-VLCM/data/tb3_lab
```

After robot identity and heading setup in the GUI, the collector previews random
goals. Press `a` to accept/start, `n` to regenerate, `f` to save the current
run as a failure, or `x` to stop collection. Dataset sampling is controlled by
`vlcm_collection.sample_rate_hz` in `config.yaml` and defaults to 30 Hz. Saved
`overhead.png` frames are cropped to the calibrated workspace and resized to
224x224 by default.

#### Option D: RL-Based Policy With Live MA-VLCM Critic Monitor
To run the same MARL policy while MA-VLCM live-plots and critiques the policy
against cumulative reward, use the MA-VLCM convenience launcher:
```bash
cd /home/adi2440/Desktop/MARL_Shahil_Aditya/MA-VLCM
export ROS_DOMAIN_ID=30
bash scripts/run_tb3_vlcm_live_monitor.sh \
  /home/adi2440/Desktop/MARL_Shahil_Aditya/MA-VLCM/checkpoints/NewFinal_0.5B.pt
```

The script starts `cv_rl_direct.launch.py`, MA-VLCM live inference on the
`/fleet_vlcm` and `/tb_N` topics, and a small live plot of predicted return
versus cumulative reward. Predictions are also logged to
`MA-VLCM/outputs/results/tb3_live_predictions.csv`.

---

### Add to Huggingface
```bash
cd /home/adi2440/Desktop/MARL_Shahil_Aditya/MA-VLCM
python scripts/upload_tb3_lab_dataset.py data/tb3_lab --repo-id adi2440/tb3-lab-vlcm
```

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

All system architecture, package structure details, configuration matrices, and setup tools are located in **[DETAILS.md](file:///home/adi2440/turtlebot_ws/DETAILS.md)**:

* 📐 **[System Architecture](file:///home/adi2440/turtlebot_ws/DETAILS.md#1-system-architecture)**: Signal flow, camera tracking pipeline, and safety loop diagram.
* 📁 **[Workspace Layout](file:///home/adi2440/turtlebot_ws/DETAILS.md#2-workspace-layout)**: Detailed outline of repository directories and internal package boundaries.
* 🧠 **[Algorithm & Reinforcement Learning Planners](file:///home/adi2440/turtlebot_ws/DETAILS.md#3-algorithm--rl-framework-deep-dive)**: Deep dive into CBS, SIPP, ORCA, and AERO-MARL graph neural network contracts.
* 🔑 **[SSH Passwordless Configuration](file:///home/adi2440/turtlebot_ws/DETAILS.md#4-ssh-setup-for-robot-bringup)**: Setting up secure, automated remote bringup.
* 📷 **[Workspace & Homography Calibration](file:///home/adi2440/turtlebot_ws/DETAILS.md#5-workspace--camera-calibration)**: How to align camera pixels to physical world coordinates.
* 🖼️ **[Static Background Capture](file:///home/adi2440/turtlebot_ws/DETAILS.md#6-one-time-background-capture)**: Setting up static background subtraction.
* ⚙️ **[Planner & Control Parameters Reference](file:///home/adi2440/turtlebot_ws/DETAILS.md#7-planning-control-parameters-reference)**: Descriptions of variables in `config.yaml`.
* 🛡️ **[Safety Envelope Guards](file:///home/adi2440/turtlebot_ws/DETAILS.md#8-safety-behavior)**: Trigger bounds and automatic shutdown triggers.
* 🩺 **[Troubleshooting & Diagnostics](file:///home/adi2440/turtlebot_ws/DETAILS.md#9-troubleshooting-guide)**: Common errors, detection adjustments, and networking resolutions.
