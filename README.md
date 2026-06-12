# TurtleBot3 CV Navigation: Scheduled Planner And RL Checkpoint Controller

This workspace runs three TurtleBot3 Burgers with a lightweight overhead-camera
localization and direct velocity-control stack. It does not use AMCL, Nav2,
map server, planner server, controller server, or lidar scans for navigation.

There are two live controller modes that use the same camera GUI, robot pose
topics, goal topics, and hardware safety envelope:

- **Scheduled planner mode**: builds an offline CBS/prioritized multi-agent
  schedule, then tracks it with a direct path follower and an ORCA-style runtime
  velocity filter.
- **RL checkpoint mode**: evaluates an AERO-MARL go-to-goal Transformer policy
  at runtime, then passes every policy action through the same conservative
  velocity, boundary, stale-pose, and inter-robot spacing checks before
  publishing `/cmd_vel`.

The shared operator workflow is:

1. Localize the robots from an overhead camera.
2. Click each robot and its initial heading in the GUI.
3. Click one goal for each robot.
4. Choose exactly one controller launch: scheduled planner or RL checkpoint.
5. Press `Enter` to start direct velocity control.
6. Publish direct velocity commands to `/tb_1/cmd_vel`, `/tb_2/cmd_vel`, and
   `/tb_3/cmd_vel`.

The old CUDA MPPI implementation is still in the workspace for comparison, but
the default conservative live controller is the scheduled planner because it
handles intersecting robot paths more predictably. The RL controller is the
experimental learned-controller path for AERO-MARL checkpoint tests.

## Algorithm References

These papers are the planning ideas this implementation follows:

- Conflict-Based Search (CBS): Sharon et al., "Conflict-Based Search For
  Optimal Multi-Agent Path Finding"  
  https://www.movingai.com/papers/sharon2015cbsjournal.html
- Safe Interval Path Planning (SIPP): Phillips and Likhachev, "SIPP: Safe
  Interval Path Planning for Dynamic Environments"  
  https://www.cs.cmu.edu/~maxim/files/sipp_icra11.pdf
- Optimal Reciprocal Collision Avoidance (ORCA): van den Berg et al.,
  "Reciprocal n-body Collision Avoidance" / ORCA project page  
  https://gamma-web.iacs.umd.edu/ORCA/
- RVO2/ORCA reference implementation page  
  https://gamma-web.iacs.umd.edu/RVO2/

In this repository, the offline planner is closer to CBS over temporal A*
states than a full SIPP implementation. The runtime local filter is
ORCA-inspired: it modifies commanded velocities near other robots and
boundaries, but it is not a full RVO2 port. The local filter can run
priority-aware, so right-of-way decisions can break symmetric deadlocks anywhere
in the workspace. A configurable central crossing gate is also available for a
known physical bottleneck, but it is disabled by default.

## RL Framework Used By The Checkpoint

The RL controller is not Nav2, not AMCL, and not a classical planner. It is a
ROS wrapper around the AERO-MARL policy checkpoint from:

```text
/home/adi2440/Desktop/MARL_Shahil_Aditya/AERO-MARL
```

The checkpoint is loaded through AERO-MARL's `TransformerPolicy` with the
`mappo_dgnn_dsgd` algorithm configuration used by
`AERO-MARL/mat/scripts/real_manygotogoal.sh`.

In practical terms, that framework is:

- **MAPPO-style multi-agent reinforcement learning**: a centralized-training,
  decentralized-execution actor-critic setup. At runtime each robot receives
  its own observation and produces its own action.
- **DGNN/communication graph policy**: the actor uses a graph neural network
  over robot-to-robot edges. The real controller builds the same fully
  populated 3-agent DGNN edge index shape used by the training contract:
  `2 x 9`.
- **Transformer policy wrapper**: AERO-MARL's `TransformerPolicy` wraps
  `MultiAgentGnnTransformer`, with the GNN encoder and decentralized continuous
  actor used for deterministic evaluation.
- **DSGD/consensus training variant**: the script name and flags use
  `mappo_dgnn_dsgd`, `truelyDistributedGNN`, `truelyDistributed`, and
  `consensusLoss`. Those are training-side choices; the ROS node only performs
  deterministic forward passes from the saved checkpoint.

For three robots, the ROS RL controller builds the same non-image go-to-goal
observation contract as `RealRobotManyGoToGoalEnv`:

- Per-agent observation dimension: `17`
- Shared critic/state dimension supplied to the policy API: `24`
- DGNN edge index shape: `2 x 9`
- Observation contents: goal distance and bearing, last command, sorted
  neighbor features, and shared robot/goal state.

The checkpoint action can include a communication-mode output when
`use_comm_action=True`. The hardware controller keeps only the first two action
values for motion:

- `raw_action[0]` becomes forward-only linear velocity:
  `tanh(raw_action[0]) -> [0, max_v_mps]`
- `raw_action[1]` becomes angular velocity:
  `tanh(raw_action[1]) -> [-max_w_radps, +max_w_radps]`
- optional `raw_action[2]` is treated as the communication mode and is not
  published to the robot.

The ROS node intentionally instantiates `TransformerPolicy` directly instead of
using AERO-MARL's `MAGoToGoalRunner`. The runner scales continuous evaluation
actions by `100`, which is appropriate for its simulator wrapper but unsafe for
hardware. The ROS controller applies its own training-compatible tanh scaling,
then clamps and filters the result with the same physical limits used by the
scheduled controller.

## Workspace Layout

Important packages:

| Package | Purpose |
| --- | --- |
| `src/cv_localization` | Overhead camera detector, EKF fusion, click GUI, fixed background, calibration files, scheduled/RL launch files, and the RL direct controller. |
| `src/multi_robot_swarm_planner` | Direct controller executable, CBS/prioritized scheduled planning, path follower, ORCA-style velocity filter, legacy CUDA MPPI code. |
| `src/multi_robot_navigation_ROS2` | SSH helper that starts namespaced robot hardware bringup on the three lab TurtleBots. |
| `src/turtlebot3`, `src/turtlebot3_msgs`, `src/DynamixelSDK` | TurtleBot3 dependencies included in the workspace. |

The scheduled direct-control launch is:

```bash
ros2 launch cv_localization cv_mppi_direct.launch.py
```

The RL checkpoint-control launch is:

```bash
ros2 launch cv_localization cv_rl_direct.launch.py \
  model_dir:=/absolute/path/to/transformer_checkpoint.pt
```

Run either the scheduled controller launch or the RL controller launch, not both
at the same time. Both publish direct velocity commands to the same
`/tb_N/cmd_vel` topics.

The scheduled launch starts only:

- `multi_robot_swarm_planner/mppi_direct_controller`
- `cv_localization/cv_mppi_direct_gui`

The RL launch starts only:

- `cv_localization/cv_rl_direct_controller`
- `cv_localization/cv_mppi_direct_gui`

Neither launch starts Nav2, AMCL, SLAM, or lidar-dependent nodes.

## Robot Inventory

Robot targets are configured in:

```text
src/multi_robot_navigation_ROS2/config/robots.yaml
```

Current lab inventory:

| Namespace | SSH login | Host | Shared domain |
| --- | --- | --- | --- |
| `tb_1` | `turtlebot@192.168.1.20` | `192.168.1.20` | `30` |
| `tb_2` | `ubuntu@192.168.1.15` | `192.168.1.15` | `30` |
| `tb_3` | `ubuntu@192.168.1.16` | `192.168.1.16` | `30` |

All robots and the control PC must use:

```bash
export ROS_DOMAIN_ID=30
```

The robot bringup helper starts `turtlebot3_bringup robot.launch.py` on each
robot with the correct namespace. The lidar may start as part of TurtleBot3
hardware bringup, but this CV navigation method does not subscribe to `/scan`.

## Fresh System Setup

These instructions assume Ubuntu 22.04 and ROS 2 Humble.

Install ROS 2 and common build/runtime dependencies:

```bash
sudo apt update
sudo apt install -y \
  ros-humble-desktop \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-opencv \
  python3-numpy \
  python3-yaml \
  openssh-client \
  build-essential \
  cmake
```

Initialize `rosdep` if this is a new ROS installation:

```bash
sudo rosdep init
rosdep update
```

Install package dependencies from the workspace:

```bash
cd ~/turtlebot_ws
rosdep install --from-paths src --ignore-src -r -y
```

The planner package is a CMake CUDA package because the legacy MPPI library is
still built. If your system has an NVIDIA GPU, install the CUDA toolkit before
building. If CUDA is installed in a nonstandard location, make sure `nvcc` is on
`PATH`.

Build and source the workspace:

```bash
cd ~/turtlebot_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

For a narrower rebuild after editing this stack:

```bash
colcon build --symlink-install \
  --packages-select cv_localization multi_robot_swarm_planner ros_multi_robot_navigation
source install/setup.bash
```

## SSH Setup For Robot Bringup

Create a key for the lab robots:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/turtlebot_lab_ed25519 -C turtlebot-lab
```

Copy the key to each robot:

```bash
ssh-copy-id -i ~/.ssh/turtlebot_lab_ed25519.pub turtlebot@192.168.1.20
ssh-copy-id -i ~/.ssh/turtlebot_lab_ed25519.pub ubuntu@192.168.1.15
ssh-copy-id -i ~/.ssh/turtlebot_lab_ed25519.pub ubuntu@192.168.1.16
```

Check that no password is requested:

```bash
ssh -i ~/.ssh/turtlebot_lab_ed25519 turtlebot@192.168.1.20 hostname
ssh -i ~/.ssh/turtlebot_lab_ed25519 ubuntu@192.168.1.15 hostname
ssh -i ~/.ssh/turtlebot_lab_ed25519 ubuntu@192.168.1.16 hostname
```

On each robot, the helper expects the robot workspace at
`~/turtlebot3_ws/install/setup.bash` if a local overlay is needed. It always
sources `/opt/ros/humble/setup.bash`.

## Camera Configuration

Edit:

```text
src/cv_localization/config/config.yaml
```

Set the overhead camera under `camera.device`. A stable `/dev/v4l/by-id/...`
path is preferred over `/dev/video0`.

List camera devices with:

```bash
ls -l /dev/v4l/by-id/
```

The configured world frame is a square workspace:

- Width: `3.048 m`
- Height: `3.048 m`
- Origin: center of the workspace
- `+x`: right in the overhead image
- `+y`: up in the overhead image
- Outer bounds: `[-1.524, +1.524] m`
- Safe goal/control bounds: outer bounds minus `wall_margin_m`

The default `wall_margin_m` is `0.20`, so robots and goals are kept at least
`0.20 m` away from the workspace boundary.

## One-Time Calibration

Calibrate the camera-to-world homography whenever the camera moves, zooms, or
changes resolution.

From the workspace root:

```bash
cd ~/turtlebot_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 src/cv_localization/cv_localization/calibrate_workspace.py \
  --config src/cv_localization/config/config.yaml \
  --output src/cv_localization/config/calibration.yaml \
  --width-m 3.048 \
  --height-m 3.048
```

Click the four workspace corners in this order:

1. Top-left
2. Top-right
3. Bottom-right
4. Bottom-left

Press `Enter` to accept. The tool shows a rectified birds-eye view before
saving `src/cv_localization/config/calibration.yaml`.

## One-Time Background Capture

The runtime GUI uses a fixed background image and never recaptures it. This is
intentional. Capture the background once with no robots in the workspace and
stable lighting.

Save it here:

```text
src/cv_localization/config/background.jpg
```

One simple capture command is:

```bash
cd ~/turtlebot_ws
python3 - <<'PY'
from pathlib import Path
import time
import cv2
import yaml

config_path = Path("src/cv_localization/config/config.yaml")
output_path = Path("src/cv_localization/config/background.jpg")
cfg = yaml.safe_load(config_path.read_text())
cam = cfg.get("camera", {})
cap = cv2.VideoCapture(cam.get("device", 0))
if not cap.isOpened():
    raise SystemExit(f"Could not open camera {cam.get('device', 0)}")
fourcc = cam.get("fourcc")
if fourcc:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.get("width", 1280))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.get("height", 960))
cap.set(cv2.CAP_PROP_FPS, cam.get("fps", 30))
for _ in range(30):
    cap.read()
    time.sleep(0.03)
ok, frame = cap.read()
cap.release()
if not ok:
    raise SystemExit("Camera read failed")
output_path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(output_path), frame)
print(f"Saved {output_path}")
PY
```

If `background.jpg` is missing, the runtime launch fails immediately with a
clear message. The GUI key `r` clears goals and planning state only; it does not
change the background.

## Planning And Control Parameters

Main parameters live in:

```text
src/cv_localization/config/config.yaml
```

Important sections:

| Parameter | Meaning |
| --- | --- |
| `mppi.control_mode` | Default is `scheduled`. Set `mppi` only for legacy MPPI comparison. The RL launch ignores this mode switch and uses the RL controller directly. |
| `mppi.planner_algorithm` | Default is `cbs`. Use `prioritized` as a faster fallback. |
| `mppi.max_v_mps` | Maximum forward linear velocity. |
| `mppi.allow_reverse` | Allows goals behind a robot to use reverse motion. |
| `mppi.max_reverse_v_mps` | Maximum reverse linear velocity. |
| `mppi.max_w_radps` | Maximum angular velocity. |
| `mppi.wall_margin_m` | Minimum distance from workspace boundaries. |
| `mppi.min_live_spacing_m` | Live inter-robot spacing below which all robots are stopped. |
| `mppi.safety_distance_m` | Minimum live robot spacing before stopping. |
| `mppi.max_dv_step` | Optional per-control-tick linear velocity slew limit. |
| `mppi.max_dw_step` | Optional per-control-tick angular velocity slew limit. |
| `mppi.planning_clearance_m` | Clearance used by the offline planner. |
| `mppi.offline_grid_resolution_m` | Spatial grid resolution for temporal A*. |
| `mppi.offline_time_step_s` | Time step for the scheduled planner. |
| `mppi.cbs_max_nodes` | CBS search node limit. |
| `mppi.path_heading_gain` | Angular gain used by the schedule follower. |
| `mppi.reverse_heading_threshold_rad` | Heading error above which reverse motion is allowed. |
| `mppi.orca_filter_enabled` | Enables the runtime ORCA-style velocity filter. |
| `mppi.orca_time_horizon_s` | Lookahead horizon for local avoidance. |
| `mppi.orca_priority_enabled` | Enables map-wide priority-aware ORCA yielding. |
| `mppi.orca_priority_wait_gain` | Increases priority for robots that have been waiting. |
| `mppi.orca_priority_schedule_lag_gain` | Increases priority for robots falling behind schedule. |
| `mppi.boundary_slowdown_margin_m` | Extra slowdown region inside the safe boundary. |

Current conservative velocity defaults are:

```yaml
mppi:
  max_v_mps: 0.1
  allow_reverse: true
  max_reverse_v_mps: 0.1
  max_w_radps: 1.0
```

If you raise velocities, test one robot first. The controller publishes
`/fleet_mppi/velocity_status` and `/tb_N/cv_measured_velocity` so you can compare
commanded and observed motion.

The RL controller reads the same `mppi.max_v_mps`, `mppi.max_w_radps`,
`mppi.wall_margin_m`, `mppi.min_live_spacing_m`, `mppi.safety_distance_m`,
`mppi.max_dv_step`, `mppi.max_dw_step`, and ORCA-style filter settings before
publishing any learned action to hardware.

## Start The Robots

Terminal A starts only the robot hardware bringup over SSH:

```bash
cd ~/turtlebot_ws
export ROS_DOMAIN_ID=30
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run ros_multi_robot_navigation bringup_lab_robots \
  --identity-file ~/.ssh/turtlebot_lab_ed25519
```

Leave this terminal running. It starts the namespaced robot nodes on the
TurtleBots.

Quick topic check from another terminal:

```bash
export ROS_DOMAIN_ID=30
source /opt/ros/humble/setup.bash
source ~/turtlebot_ws/install/setup.bash
ros2 topic list | grep -E '/tb_[123]/(odom|cmd_vel)'
```

You should see `/tb_1/odom`, `/tb_2/odom`, `/tb_3/odom`, and the command topics.
No `/scan` topic is required for this method.

## Start CV Scheduled Navigation

Terminal B starts the camera GUI and scheduled direct controller:

```bash
cd ~/turtlebot_ws
export ROS_DOMAIN_ID=30
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch cv_localization cv_mppi_direct.launch.py
```

Optional explicit file paths:

```bash
ros2 launch cv_localization cv_mppi_direct.launch.py \
  config:=/home/adi2440/turtlebot_ws/src/cv_localization/config/config.yaml \
  calibration:=/home/adi2440/turtlebot_ws/src/cv_localization/config/calibration.yaml \
  background:=/home/adi2440/turtlebot_ws/src/cv_localization/config/background.jpg
```

The launch sanitizes `LD_LIBRARY_PATH` to avoid common Anaconda `libstdc++`
conflicts and sets `TURTLEBOT3_MODEL=burger`.

## Start CV RL Checkpoint Navigation

Use this instead of `cv_mppi_direct.launch.py` when the AERO-MARL checkpoint is
the main controller. The GUI workflow is the same: click robot identities,
click headings, click goals, then press `Enter` to start and `Space`/`Esc` to
stop.

```bash
cd ~/turtlebot_ws
export ROS_DOMAIN_ID=30
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch cv_localization cv_rl_direct.launch.py \
  model_dir:=/home/adi2440/Desktop/MARL_Shahil_Aditya/AERO-MARL/transformer_800.pt \
  aero_marl_root:=/home/adi2440/Desktop/MARL_Shahil_Aditya/AERO-MARL
```

The default `model_dir` mirrors the checkpoint path from
`AERO-MARL/mat/scripts/real_manygotogoal.sh`, but that path is not present on
this machine. Pass the checkpoint explicitly; the RL node exits with a clear
error if the file is missing.

The RL controller subscribes to the same `/tb_N/cv_pose` and `/tb_N/mppi_goal`
topics as the scheduled controller and preserves the configured safety limits:
velocity clamps, safe workspace boundary, stale-pose stop, live inter-robot
spacing stop, boundary slowdown, and ORCA-style local velocity filtering.

## GUI Workflow

The GUI window is named `CV Scheduled Multi-Agent Control`.
Robot overlays are color-coded as `tb_1` red, `tb_2` blue, and `tb_3` green.

1. Wait until the three robot blobs are visible.
2. Click the robot blob for `tb_1`.
3. Click a nearby point in the direction `tb_1` is facing.
4. Repeat robot blob plus heading clicks for `tb_2` and `tb_3`.
5. Click goal locations in order: `tb_1`, `tb_2`, `tb_3`.
6. In scheduled mode, wait for the offline scheduled paths to appear. In RL
   mode, verify the current-goal preview/status appears and the controller is
   ready.
7. Press `Enter` to start motion.

Keyboard controls:

| Key | Action |
| --- | --- |
| `Enter` | Start direct velocity control after all goals are set. |
| `Space` | Stop all robots immediately. |
| `Esc` | Stop all robots immediately. |
| `r` | Clear goals and planning state. The fixed background is unchanged. |
| `q` | Stop all robots and quit the GUI. |

The initial heading click matters because odometry yaw is not reliable
immediately at startup. After the click, the GUI runs a small EKF:

- Camera supplies `x/y`.
- The clicked heading supplies initial yaw.
- `/tb_N/odom` supplies yaw deltas.

Goal yaw is ignored. Goals are position-only.

The scheduled controller uses priority-aware ORCA by default. During a local
conflict, the lower-priority robot takes more of the avoidance correction. A
robot that has been waiting or has fallen behind its schedule gains priority
over time, so right-of-way can rotate instead of locking to one robot forever.
An optional central crossing gate can still be enabled under `mppi.crossing_*`
for a known physical bottleneck.

The RL controller uses the same GUI start, stop, and clear services. Before
`Enter`, it receives poses and goals but publishes zero velocity. After
`Enter`, each policy action is scaled, clamped, slew-limited, and filtered for
boundary and inter-robot safety before being published.

## ROS Topics And Services

Published or consumed topics:

| Topic | Purpose |
| --- | --- |
| `/tb_N/cv_pose` | Fused camera/EKF robot pose from the GUI to the controller. |
| `/tb_N/odom` | Robot odometry yaw deltas for the EKF and velocity reporting. |
| `/tb_N/mppi_goal` | Goal point topic. Name is legacy; scheduled planning uses it too. |
| `/tb_N/offline_plan` | Offline CBS/prioritized scheduled path preview. The RL controller does not publish an offline schedule. |
| `/tb_N/mppi_plan` | Remaining/current tracked path preview in scheduled mode, or current-goal preview in RL mode. Name is legacy. |
| `/tb_N/cmd_vel` | Direct velocity command sent to each robot. |
| `/tb_N/cv_measured_velocity` | Measured velocity estimate from CV/odom. |
| `/fleet_mppi/status` | Fleet controller status messages. |
| `/fleet_mppi/velocity_status` | Periodic commanded/observed velocity summary. |

Services:

| Service | Purpose |
| --- | --- |
| `/fleet_mppi/start` | Start motion after goals and paths are ready. |
| `/fleet_mppi/stop` | Publish zero velocity and stop all robots. |
| `/fleet_mppi/clear_goals` | Clear goals and stop. |
| `/fleet_mppi/plan` | Recompute the offline schedule in scheduled mode; validate goals/current preview in RL mode. |

Manual stop command:

```bash
ros2 service call /fleet_mppi/stop std_srvs/srv/Trigger {}
```

Check velocity status:

```bash
ros2 topic echo /fleet_mppi/velocity_status
```

Watch command velocity for one robot:

```bash
ros2 topic echo /tb_1/cmd_vel
```

## Recreating A Navigation Run From Scratch

Use this checklist on a new machine or after moving the camera:

1. Install ROS 2 Humble and dependencies.
2. Clone or copy this workspace into `~/turtlebot_ws`.
3. Run `rosdep install --from-paths src --ignore-src -r -y`.
4. Build with `colcon build --symlink-install`.
5. Configure SSH access to all three robots.
6. Confirm all robots are on the same WiFi and reachable by SSH.
7. Set `ROS_DOMAIN_ID=30`.
8. Update `src/cv_localization/config/config.yaml` with the overhead camera path.
9. Run the workspace calibration tool and save `calibration.yaml`.
10. Capture `background.jpg` once with no robots in the workspace.
11. Start robot hardware bringup with `bringup_lab_robots`.
12. Start exactly one live controller launch:
    `cv_mppi_direct.launch.py` for scheduled planning or
    `cv_rl_direct.launch.py` for the RL checkpoint.
13. Click robot identities and headings in order.
14. Click goals in order.
15. Verify the preview paths or RL current-goal previews stay inside the safe
    boundary.
16. Press `Enter` to move.
17. Use `Space` or `Esc` as the emergency stop.

## Safety Behavior

Both live controllers publish zero velocity to all robots when any of these
occur:

- `Space`, `Esc`, `q`, or `/fleet_mppi/stop`
- Missing or stale localization
- Missing robot pose
- Unsafe live robot spacing
- Boundary violation risk
- Goal completion
- Node shutdown

Goals outside the safe boundary are rejected in the GUI. The safe boundary is
drawn on screen.

In RL mode, this safety layer runs after policy inference and before
`/tb_N/cmd_vel`. A bad or surprising checkpoint output is still limited by
`max_v_mps`, `max_w_radps`, optional slew limits, stale-pose checks, live
spacing checks, boundary slowdown, and the ORCA-style local velocity filter.

## Dry Checks Before Live Motion

Build the relevant packages:

```bash
cd ~/turtlebot_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install \
  --packages-select cv_localization multi_robot_swarm_planner ros_multi_robot_navigation
source install/setup.bash
```

Confirm the direct launch does not include Nav2 or AMCL:

```bash
ros2 launch cv_localization cv_mppi_direct.launch.py --show-args
```

Confirm the RL launch arguments and checkpoint path before using the learned
controller:

```bash
ros2 launch cv_localization cv_rl_direct.launch.py --show-args
```

Dry-run the RL node long enough to validate imports and checkpoint loading:

```bash
timeout 8 ros2 run cv_localization cv_rl_direct_controller --ros-args \
  -p model_dir:=/home/adi2440/Desktop/MARL_Shahil_Aditya/AERO-MARL/transformer_800.pt \
  -p aero_marl_root:=/home/adi2440/Desktop/MARL_Shahil_Aditya/AERO-MARL
```

The timeout command should end because of the timeout, not because of a Python
traceback. If the checkpoint is missing, the node should fail immediately with
a clear `model_dir` error.

In another terminal, inspect nodes after launch:

```bash
ros2 node list
```

Expected navigation nodes are the GUI plus either `mppi_direct_controller` or
`cv_rl_direct_controller`. You should not see `amcl`, `planner_server`,
`controller_server`, `bt_navigator`, `map_server`, or other Nav2 lifecycle
nodes from these launches.

Before pressing `Enter`, verify no live command velocity is being sent:

```bash
ros2 topic echo /tb_1/cmd_vel
```

After pressing `Space` or `Esc`, the latest command should be zero on all three
robots.

## Troubleshooting

If the GUI fails with a missing background error, create:

```text
src/cv_localization/config/background.jpg
```

If the RL launch exits before the GUI can start or plan:

- Check that `model_dir` points to an existing checkpoint such as
  `/home/adi2440/Desktop/MARL_Shahil_Aditya/AERO-MARL/transformer_800.pt`.
- Check that `aero_marl_root` points to the AERO-MARL repository root, not the
  `mat` subdirectory.
- Source the ROS workspace after rebuilding:
  `source ~/turtlebot_ws/install/setup.bash`.
- Partial-restore log messages can be normal when `allow_partial_restore=True`
  adapts a checkpoint to the current robot count or observation shape.

If the GUI says `Cannot plan: service is not ready yet` or
`Cannot start: service is not ready yet`, the controller process is probably not
running. Check the launch terminal for a Python traceback or missing checkpoint
message.

If detections are unstable:

- Keep lighting fixed after capturing `background.jpg`.
- Remove robots and recapture `background.jpg` only during setup.
- Tune `detection.background_diff_threshold`.
- Tune `detection.robot_dark_threshold`.
- Tune `detection.min_blob_area` and `detection.max_blob_area`.
- If identities swap during close passes, tune `tracking.position_history_size`,
  `tracking.prediction_horizon_sec`, and `tracking.prediction_weight`.

If robots do not turn enough:

- Increase `mppi.max_w_radps`.
- Increase `mppi.path_heading_gain`.
- Keep `mppi.reverse_heading_threshold_rad` near `2.20` so side goals rotate
  instead of immediately choosing reverse.
- Check that the initial heading click is accurate.

If robots deadlock at the central crossing:

- Confirm `mppi.orca_priority_enabled: true`.
- Increase `mppi.orca_priority_wait_gain` if waiting robots should win sooner.
- Increase `mppi.orca_priority_strength` if the lower-priority robot should
  yield more decisively.
- Enable `mppi.crossing_gate_enabled` only if you want a fixed bottleneck token.

If a scheduled-mode robot will not drive to a goal behind it:

- Confirm `mppi.allow_reverse: true`.
- Check `mppi.max_reverse_v_mps`.
- Verify the goal is inside the safe boundary.
- Watch `/tb_N/cmd_vel` to confirm negative `linear.x` is allowed.

In RL mode, the hardware adapter intentionally maps the learned linear action
to forward-only velocity. A goal behind the robot should cause turning first,
not negative `linear.x`.

If robots appear faster than expected:

- Check `mppi.max_v_mps`, `mppi.max_reverse_v_mps`, and `mppi.max_w_radps`.
- Echo `/fleet_mppi/velocity_status`.
- Echo `/tb_N/cv_measured_velocity`.
- Confirm the camera calibration scale is correct. A bad homography can make
  measured speeds look wrong.

If SSH bringup fails:

```bash
ping -c 3 192.168.1.20
ping -c 3 192.168.1.15
ping -c 3 192.168.1.16
ssh -i ~/.ssh/turtlebot_lab_ed25519 turtlebot@192.168.1.20 hostname
ssh -i ~/.ssh/turtlebot_lab_ed25519 ubuntu@192.168.1.15 hostname
ssh -i ~/.ssh/turtlebot_lab_ed25519 ubuntu@192.168.1.16 hostname
```

If ROS topics are missing:

- Confirm the PC and robots use `ROS_DOMAIN_ID=30`.
- Confirm `ROS_LOCALHOST_ONLY` is unset or `0`.
- Confirm the robot-side bringup terminal is still running.
- Run one robot at a time with `--robot tb_1`, `--robot tb_2`, or `--robot tb_3`
  to isolate hardware issues.

## What Not To Launch

For this method, do not launch:

- `multi_nav2_launch.py`
- `lab_three_robot_nav.launch.py`
- AMCL
- Nav2 planner/controller/BT navigator
- `cv_mppi_direct.launch.py` and `cv_rl_direct.launch.py` at the same time
- Map server
- SLAM Toolbox
- Any lidar-dependent navigation node

Those files remain in the workspace for legacy experiments, but they are not
part of the current CV direct-control workflow.
