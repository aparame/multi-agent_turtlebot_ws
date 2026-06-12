#include "multi_robot_swarm_planner/direct_room_mppi.hpp"

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/utils.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std::chrono_literals;

namespace multi_robot_swarm_planner
{
namespace
{

std::string trim(const std::string & text)
{
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1);
}

std::vector<std::string> splitCsv(const std::string & csv)
{
  std::vector<std::string> values;
  std::stringstream stream(csv);
  std::string item;
  while (std::getline(stream, item, ',')) {
    item = trim(item);
    if (!item.empty()) {
      values.push_back(item);
    }
  }
  return values;
}

double distance2d(const DirectMppiState & a, const DirectMppiState & b)
{
  const double dx = static_cast<double>(a.x - b.x);
  const double dy = static_cast<double>(a.y - b.y);
  return std::sqrt(dx * dx + dy * dy);
}

double normalizeAngle(double angle)
{
  while (angle > M_PI) {
    angle -= 2.0 * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }
  return angle;
}

}  // namespace

class MppiDirectController : public rclcpp::Node
{
public:
  MppiDirectController()
  : Node("mppi_direct_controller")
  {
    robot_names_ = splitCsv(this->declare_parameter<std::string>(
      "robot_names_csv", "tb_1,tb_2,tb_3"));
    if (robot_names_.empty()) {
      throw std::runtime_error("mppi_direct_controller needs at least one robot");
    }

    map_frame_ = this->declare_parameter<std::string>("map_frame", "map");
    control_rate_hz_ = this->declare_parameter<double>("control_rate_hz", 10.0);
    pose_timeout_s_ = this->declare_parameter<double>("pose_timeout_s", 0.5);
    min_live_spacing_m_ = this->declare_parameter<double>("min_live_spacing_m", 0.22);
    control_mode_ = this->declare_parameter<std::string>("control_mode", "scheduled");
    planner_algorithm_ = this->declare_parameter<std::string>("planner_algorithm", "cbs");

    planner_params_.dt = static_cast<float>(this->declare_parameter<double>("dt", 0.10));
    planner_params_.room_size_m = static_cast<float>(
      this->declare_parameter<double>("room_size_m", 3.048));
    planner_params_.goal_radius_m = static_cast<float>(
      this->declare_parameter<double>("goal_radius_m", 0.12));
    planner_params_.max_v_mps = static_cast<float>(
      this->declare_parameter<double>("max_v_mps", 0.50));
    allow_reverse_ = this->declare_parameter<bool>("allow_reverse", true);
    max_reverse_v_mps_ = this->declare_parameter<double>(
      "max_reverse_v_mps", static_cast<double>(planner_params_.max_v_mps));
    planner_params_.max_w_radps = static_cast<float>(
      this->declare_parameter<double>("max_w_radps", 0.20));
    planner_params_.max_dv_step = static_cast<float>(
      this->declare_parameter<double>("max_dv_step", 0.03));
    planner_params_.max_dw_step = static_cast<float>(
      this->declare_parameter<double>("max_dw_step", 0.18));
    planner_params_.safety_distance_m = static_cast<float>(
      this->declare_parameter<double>("safety_distance_m", 0.25));
    planner_params_.planning_clearance_m = static_cast<float>(
      this->declare_parameter<double>("planning_clearance_m", 0.45));
    planner_params_.wall_margin_m = static_cast<float>(
      this->declare_parameter<double>("wall_margin_m", 0.20));
    planner_params_.horizon = this->declare_parameter<int>("horizon", 80);
    planner_params_.samples = this->declare_parameter<int>("samples", 512);
    planner_params_.mppi_iterations = this->declare_parameter<int>("mppi_iterations", 6);
    offline_planner_enabled_ = this->declare_parameter<bool>("offline_planner_enabled", true);
    offline_grid_resolution_m_ = this->declare_parameter<double>("offline_grid_resolution_m", 0.10);
    offline_time_step_s_ = this->declare_parameter<double>("offline_time_step_s", 0.50);
    offline_max_time_steps_ = this->declare_parameter<int>("offline_max_time_steps", 180);
    offline_wait_penalty_ = this->declare_parameter<double>("offline_wait_penalty", 0.05);
    cbs_max_nodes_ = this->declare_parameter<int>("cbs_max_nodes", 512);
    waypoint_reached_m_ = this->declare_parameter<double>("waypoint_reached_m", 0.14);
    waypoint_lookahead_m_ = this->declare_parameter<double>("waypoint_lookahead_m", 0.30);
    path_heading_gain_ = this->declare_parameter<double>("path_heading_gain", 2.0);
    reverse_heading_threshold_rad_ =
      this->declare_parameter<double>("reverse_heading_threshold_rad", 2.20);
    path_slow_heading_rad_ = this->declare_parameter<double>("path_slow_heading_rad", 0.70);
    path_stop_heading_rad_ = this->declare_parameter<double>("path_stop_heading_rad", 1.40);
    path_goal_slowdown_m_ = this->declare_parameter<double>("path_goal_slowdown_m", 0.35);
    orca_filter_enabled_ = this->declare_parameter<bool>("orca_filter_enabled", true);
    orca_radius_m_ = this->declare_parameter<double>("orca_radius_m", 0.16);
    orca_neighbor_dist_m_ = this->declare_parameter<double>("orca_neighbor_dist_m", 0.75);
    orca_time_horizon_s_ = this->declare_parameter<double>("orca_time_horizon_s", 2.0);
    orca_avoidance_gain_ = this->declare_parameter<double>("orca_avoidance_gain", 1.0);
    orca_priority_enabled_ = this->declare_parameter<bool>("orca_priority_enabled", true);
    orca_priority_strength_ = this->declare_parameter<double>("orca_priority_strength", 2.0);
    orca_priority_fixed_bias_ = this->declare_parameter<double>("orca_priority_fixed_bias", 0.25);
    orca_priority_wait_gain_ = this->declare_parameter<double>("orca_priority_wait_gain", 0.6);
    orca_priority_schedule_lag_gain_ =
      this->declare_parameter<double>("orca_priority_schedule_lag_gain", 0.2);
    orca_priority_min_share_ = this->declare_parameter<double>("orca_priority_min_share", 0.10);
    orca_priority_wait_speed_mps_ =
      this->declare_parameter<double>("orca_priority_wait_speed_mps", 0.02);
    orca_priority_wait_decay_gain_ =
      this->declare_parameter<double>("orca_priority_wait_decay_gain", 2.0);
    boundary_slowdown_margin_m_ =
      this->declare_parameter<double>("boundary_slowdown_margin_m", 0.15);
    crossing_gate_enabled_ = this->declare_parameter<bool>("crossing_gate_enabled", false);
    crossing_center_x_m_ = this->declare_parameter<double>("crossing_center_x_m", 0.0);
    crossing_center_y_m_ = this->declare_parameter<double>("crossing_center_y_m", 0.0);
    crossing_zone_radius_m_ = this->declare_parameter<double>("crossing_zone_radius_m", 0.36);
    crossing_entry_radius_m_ = this->declare_parameter<double>("crossing_entry_radius_m", 0.62);
    crossing_release_radius_m_ =
      this->declare_parameter<double>("crossing_release_radius_m", 0.50);
    crossing_progress_timeout_s_ =
      this->declare_parameter<double>("crossing_progress_timeout_s", 4.0);
    crossing_progress_epsilon_m_ =
      this->declare_parameter<double>("crossing_progress_epsilon_m", 0.03);
    velocity_status_period_s_ = this->declare_parameter<double>("velocity_status_period_s", 1.0);

    status_pub_ = this->create_publisher<std_msgs::msg::String>("/fleet_mppi/status", 10);
    velocity_status_pub_ =
      this->create_publisher<std_msgs::msg::String>("/fleet_mppi/velocity_status", 10);

    for (const auto & robot : robot_names_) {
      auto & state = robots_[robot];
      state.cmd_pub = this->create_publisher<geometry_msgs::msg::Twist>(
        "/" + robot + "/cmd_vel", 10);
      state.actual_velocity_pub = this->create_publisher<geometry_msgs::msg::Twist>(
        "/" + robot + "/cv_measured_velocity", 10);
      state.path_pub = this->create_publisher<nav_msgs::msg::Path>(
        "/" + robot + "/mppi_plan", 10);
      state.offline_path_pub = this->create_publisher<nav_msgs::msg::Path>(
        "/" + robot + "/offline_plan", 10);

      pose_subs_.push_back(this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/" + robot + "/cv_pose",
        10,
        [this, robot](geometry_msgs::msg::PoseStamped::SharedPtr msg) {
          handlePose(robot, *msg);
        }));
      goal_subs_.push_back(this->create_subscription<geometry_msgs::msg::PointStamped>(
        "/" + robot + "/mppi_goal",
        10,
        [this, robot](geometry_msgs::msg::PointStamped::SharedPtr msg) {
          handleGoal(robot, *msg);
        }));
      odom_subs_.push_back(this->create_subscription<nav_msgs::msg::Odometry>(
        "/" + robot + "/odom",
        10,
        [this, robot](nav_msgs::msg::Odometry::SharedPtr msg) {
          handleOdom(robot, *msg);
        }));
    }

    start_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "/fleet_mppi/start",
      [this](
        const std::shared_ptr<std_srvs::srv::Trigger::Request>,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        std::string reason;
        if (!readyToRun(reason)) {
          response->success = false;
          response->message = reason;
          publishStatus("cannot start: " + reason, true);
          return;
        }
        active_ = true;
        schedule_start_time_ = this->now();
        resetCrossingGate();
        plan_requested_ = offlinePathsMissing();
        response->success = true;
        response->message = "scheduled multi-agent control started";
        publishStatus(response->message, true);
      });

    stop_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "/fleet_mppi/stop",
      [this](
        const std::shared_ptr<std_srvs::srv::Trigger::Request>,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        stopAll("stop service");
        response->success = true;
        response->message = "stopped all robots";
      });

    clear_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "/fleet_mppi/clear_goals",
      [this](
        const std::shared_ptr<std_srvs::srv::Trigger::Request>,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        stopAll("clear goals");
        for (auto & entry : robots_) {
          entry.second.has_goal = false;
          entry.second.offline_path.clear();
          entry.second.waypoint_index = 0;
          publishPath(entry.first, {});
          publishOfflinePath(entry.first, {});
        }
        plan_requested_ = false;
        response->success = true;
        schedule_start_time_.reset();
        resetCrossingGate();
        response->message = "cleared scheduled multi-agent goals";
        publishStatus(response->message, true);
      });

    plan_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "/fleet_mppi/plan",
      [this](
        const std::shared_ptr<std_srvs::srv::Trigger::Request>,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        std::string reason;
        if (!readyToPlan(reason)) {
          response->success = false;
          response->message = reason;
          publishStatus("cannot plan: " + reason, true);
          return;
        }
        plan_requested_ = true;
        response->success = true;
        response->message = "scheduled multi-agent preview requested";
      });

    const double safe_rate = std::max(1.0, control_rate_hz_);
    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(1.0 / safe_rate),
      [this]() {
        timerCallback();
      });

    std::ostringstream ready;
    ready << "direct controller ready; no AMCL/Nav2/scan subscriptions; mode="
          << control_mode_ << ", planner=" << planner_algorithm_ << "; limits v<="
          << planner_params_.max_v_mps << " m/s, w<=" << planner_params_.max_w_radps
          << " rad/s, reverse=" << (allow_reverse_ ? "enabled" : "disabled")
          << "; scheduled planning "
          << (offline_planner_enabled_ ? "enabled" : "disabled");
    publishStatus(ready.str(), true);
  }

  void stopForShutdown()
  {
    stopAll("shutdown");
  }

private:
  struct WorldVelocity
  {
    double x{0.0};
    double y{0.0};
  };

  struct RobotRuntime
  {
    DirectMppiState pose;
    DirectMppiState previous_pose;
    DirectMppiState goal;
    DirectMppiControl last_control;
    DirectMppiControl measured_cv_velocity;
    DirectMppiControl measured_odom_velocity;
    float measured_cv_lateral_mps{0.0f};
    float measured_cv_speed_mps{0.0f};
    double orca_priority_wait_s{0.0};
    std::vector<DirectMppiState> offline_path;
    size_t waypoint_index{0};
    bool has_pose{false};
    bool has_previous_pose{false};
    bool has_goal{false};
    bool has_odom_velocity{false};
    std::optional<rclcpp::Time> pose_time;
    std::optional<rclcpp::Time> previous_pose_time;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr actual_velocity_pub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr offline_path_pub;
  };

  void handlePose(const std::string & robot, const geometry_msgs::msg::PoseStamped & msg)
  {
    auto & state = robots_[robot];
    const auto now = this->now();
    if (state.has_pose && state.pose_time) {
      state.previous_pose = state.pose;
      state.previous_pose_time = state.pose_time;
      state.has_previous_pose = true;
    }
    state.pose.x = static_cast<float>(msg.pose.position.x);
    state.pose.y = static_cast<float>(msg.pose.position.y);
    state.pose.theta = static_cast<float>(tf2::getYaw(msg.pose.orientation));
    state.pose_time = now;
    state.has_pose = true;

    if (state.has_previous_pose && state.previous_pose_time) {
      const double dt = (now - *state.previous_pose_time).seconds();
      if (dt > 1e-3) {
        const double dx = static_cast<double>(state.pose.x - state.previous_pose.x);
        const double dy = static_cast<double>(state.pose.y - state.previous_pose.y);
        const double vx_world = dx / dt;
        const double vy_world = dy / dt;
        const double heading = static_cast<double>(state.pose.theta);
        state.measured_cv_velocity.v = static_cast<float>(
          vx_world * std::cos(heading) + vy_world * std::sin(heading));
        state.measured_cv_lateral_mps = static_cast<float>(
          -vx_world * std::sin(heading) + vy_world * std::cos(heading));
        state.measured_cv_speed_mps = static_cast<float>(std::hypot(vx_world, vy_world));
        state.measured_cv_velocity.omega = static_cast<float>(
          normalizeAngle(state.pose.theta - state.previous_pose.theta) / dt);
        publishMeasuredVelocity(state);
      }
    }
  }

  void handleOdom(const std::string & robot, const nav_msgs::msg::Odometry & msg)
  {
    auto & state = robots_[robot];
    state.measured_odom_velocity.v = static_cast<float>(msg.twist.twist.linear.x);
    state.measured_odom_velocity.omega = static_cast<float>(msg.twist.twist.angular.z);
    state.has_odom_velocity = true;
  }

  void handleGoal(const std::string & robot, const geometry_msgs::msg::PointStamped & msg)
  {
    if (!msg.header.frame_id.empty() && msg.header.frame_id != map_frame_) {
      publishStatus(
        "accepted " + robot + " goal without TF transform; expected frame " + map_frame_,
        true);
    }
    auto & state = robots_[robot];
    DirectMppiState goal;
    goal.x = static_cast<float>(msg.point.x);
    goal.y = static_cast<float>(msg.point.y);
    goal.theta = 0.0f;
    std::string reason;
    if (!insideSafeBoundary(goal, reason)) {
      state.has_goal = false;
      publishStatus("rejected " + robot + " goal: " + reason, true);
      return;
    }
    const bool changed_goal = !state.has_goal || distance2d(goal, state.goal) > 0.01;
    state.goal = goal;
    state.has_goal = true;
    if (changed_goal) {
      state.offline_path.clear();
      state.waypoint_index = 0;
      state.orca_priority_wait_s = 0.0;
      plan_requested_ = true;
      resetCrossingGate();
      publishStatus(
        "stored " + robot + " direct goal x=" + std::to_string(msg.point.x) +
        " y=" + std::to_string(msg.point.y),
        true);
    }
  }

  void timerCallback()
  {
    if (!allGoalsReady()) {
      return;
    }

    std::string reason;
    if (!readyToPlan(reason)) {
      if (active_) {
        stopAll(reason);
      } else {
        publishStatus("waiting to preview: " + reason);
      }
      return;
    }

    if (!liveSpacingSafe(reason)) {
      stopAll(reason);
      return;
    }

    if (offline_planner_enabled_ && (plan_requested_ || offlinePathsMissing())) {
      if (!planOfflinePaths(reason)) {
        if (active_) {
          stopAll("scheduled planner failure: " + reason);
        } else {
          publishStatus("scheduled planner failed: " + reason, true);
        }
        return;
      }
      publishStatus("scheduled multi-agent paths ready", true);
    }

    if (control_mode_ == "mppi") {
      runMppiStep();
      return;
    }
    if (control_mode_ == "scheduled") {
      runScheduledStep();
      return;
    }

    const std::string mode_error = "unknown control_mode '" + control_mode_ + "'";
    if (active_) {
      stopAll(mode_error);
    } else {
      publishStatus(mode_error, true);
    }
  }

  void runMppiStep()
  {
    std::vector<DirectMppiState> states;
    std::vector<DirectMppiState> local_goals;
    std::vector<DirectMppiControl> last_controls;
    for (const auto & robot : robot_names_) {
      auto & runtime = robots_.at(robot);
      states.push_back(runtime.pose);
      local_goals.push_back(selectLocalGoal(runtime));
      last_controls.push_back(active_ ? runtime.last_control : DirectMppiControl{0.0f, 0.0f});
    }

    DirectMppiResult result = computeDirectRoomMppi(
      states, local_goals, last_controls, planner_params_);
    if (!result.success) {
      if (active_) {
        stopAll("MPPI failure: " + result.message);
      } else {
        publishStatus("MPPI preview failed: " + result.message);
      }
      return;
    }

    for (size_t i = 0; i < robot_names_.size(); ++i) {
      publishPath(robot_names_[i], result.preview_paths[i]);
    }

    if (finalGoalsReached()) {
      if (active_) {
        stopAll("all goals reached");
      }
      publishStatus("all direct MPPI goals reached");
      return;
    }

    if (!active_) {
      publishStatus("direct MPPI preview ready");
      return;
    }

    for (size_t i = 0; i < robot_names_.size(); ++i) {
      auto & runtime = robots_.at(robot_names_[i]);
      runtime.last_control = publishControl(runtime, result.controls[i]);
    }
    publishVelocityStatus();
  }

  void runScheduledStep()
  {
    publishScheduledPreviewPaths();

    if (finalGoalsReached()) {
      if (active_) {
        stopAll("all goals reached");
      }
      publishStatus("all scheduled goals reached");
      return;
    }

    if (!active_) {
      publishStatus("scheduled preview ready");
      return;
    }
    if (!schedule_start_time_) {
      schedule_start_time_ = this->now();
    }

    std::vector<WorldVelocity> world_velocities;
    std::vector<DirectMppiControl> preferred_controls;
    world_velocities.reserve(robot_names_.size());
    preferred_controls.reserve(robot_names_.size());
    for (const auto & robot : robot_names_) {
      auto & runtime = robots_.at(robot);
      const DirectMppiControl preferred = computeScheduledControl(runtime);
      preferred_controls.push_back(preferred);
      world_velocities.push_back(controlToWorld(runtime.pose, preferred));
    }

    const std::vector<bool> crossing_gate_holds =
      applyCrossingGate(preferred_controls, world_velocities);
    updateOrcaPriorityAging(preferred_controls, crossing_gate_holds);
    const std::vector<double> orca_priorities = computeOrcaPriorityScores();
    if (orca_filter_enabled_) {
      applyOrcaLikeFilter(world_velocities, crossing_gate_holds, orca_priorities);
    }
    applyBoundaryVelocityFilter(world_velocities);

    for (size_t i = 0; i < robot_names_.size(); ++i) {
      auto & runtime = robots_.at(robot_names_[i]);
      const DirectMppiControl filtered = mergeFilteredLinearWithPreferredTurn(
        runtime.pose,
        preferred_controls[i],
        world_velocities[i]);
      runtime.last_control = publishControl(runtime, filtered);
    }
    publishVelocityStatus();
  }

  double effectiveOfflineTimeStep() const
  {
    const double max_v = std::max(1e-3, static_cast<double>(planner_params_.max_v_mps));
    return std::max(offline_time_step_s_, offline_grid_resolution_m_ / max_v);
  }

  int currentScheduleIndex() const
  {
    if (!schedule_start_time_) {
      return 0;
    }
    const double elapsed = std::max(0.0, (this->now() - *schedule_start_time_).seconds());
    return static_cast<int>(std::floor(elapsed / effectiveOfflineTimeStep()));
  }

  size_t scheduledTargetIndex(RobotRuntime & runtime)
  {
    if (runtime.offline_path.empty()) {
      return 0;
    }
    while (runtime.waypoint_index + 1 < runtime.offline_path.size() &&
      distance2d(runtime.pose, runtime.offline_path[runtime.waypoint_index]) < waypoint_reached_m_)
    {
      ++runtime.waypoint_index;
    }
    const size_t schedule_index = static_cast<size_t>(
      std::clamp(currentScheduleIndex(), 0, static_cast<int>(runtime.offline_path.size()) - 1));
    return std::max(runtime.waypoint_index, schedule_index);
  }

  DirectMppiControl computeScheduledControl(RobotRuntime & runtime)
  {
    if (runtime.offline_path.empty()) {
      return computePointTrackingControl(runtime.pose, runtime.goal);
    }
    const size_t target_index = scheduledTargetIndex(runtime);
    const DirectMppiState target =
      runtime.offline_path[std::min(target_index, runtime.offline_path.size() - 1)];
    return computePointTrackingControl(runtime.pose, target);
  }

  DirectMppiControl computePointTrackingControl(
    const DirectMppiState & pose,
    const DirectMppiState & target) const
  {
    const double dx = static_cast<double>(target.x - pose.x);
    const double dy = static_cast<double>(target.y - pose.y);
    const double dist = std::hypot(dx, dy);

    DirectMppiControl control;
    if (dist < planner_params_.goal_radius_m * 0.5) {
      return control;
    }

    const double desired_heading = std::atan2(dy, dx);
    double heading_error = normalizeAngle(desired_heading - pose.theta);
    bool reverse = false;
    if (allow_reverse_ && std::abs(heading_error) > reverse_heading_threshold_rad_) {
      reverse = true;
      heading_error = normalizeAngle(desired_heading + M_PI - pose.theta);
    }

    const double speed_limit = reverse ?
      std::max(0.0, max_reverse_v_mps_) :
      static_cast<double>(planner_params_.max_v_mps);
    double speed = speed_limit;
    speed *= std::clamp(dist / std::max(0.05, path_goal_slowdown_m_), 0.0, 1.0);
    if (std::abs(heading_error) > path_slow_heading_rad_) {
      speed *= 0.35;
    }
    if (std::abs(heading_error) > path_stop_heading_rad_) {
      speed = 0.0;
    }

    control.v = static_cast<float>(reverse ? -speed : speed);
    control.omega = static_cast<float>(std::clamp(
      path_heading_gain_ * heading_error,
      -static_cast<double>(planner_params_.max_w_radps),
      static_cast<double>(planner_params_.max_w_radps)));
    return control;
  }

  WorldVelocity controlToWorld(
    const DirectMppiState & pose,
    const DirectMppiControl & control) const
  {
    return WorldVelocity{
      static_cast<double>(control.v) * std::cos(static_cast<double>(pose.theta)),
      static_cast<double>(control.v) * std::sin(static_cast<double>(pose.theta))};
  }

  double distanceToCrossing(const DirectMppiState & pose) const
  {
    const double dx = static_cast<double>(pose.x) - crossing_center_x_m_;
    const double dy = static_cast<double>(pose.y) - crossing_center_y_m_;
    return std::hypot(dx, dy);
  }

  double distanceToCrossingSegment(
    const DirectMppiState & a,
    const DirectMppiState & b) const
  {
    const double ax = static_cast<double>(a.x);
    const double ay = static_cast<double>(a.y);
    const double bx = static_cast<double>(b.x);
    const double by = static_cast<double>(b.y);
    const double vx = bx - ax;
    const double vy = by - ay;
    const double len2 = vx * vx + vy * vy;
    if (len2 < 1e-9) {
      return distanceToCrossing(a);
    }
    const double t = std::clamp(
      ((crossing_center_x_m_ - ax) * vx + (crossing_center_y_m_ - ay) * vy) / len2,
      0.0,
      1.0);
    const double px = ax + t * vx;
    const double py = ay + t * vy;
    return std::hypot(px - crossing_center_x_m_, py - crossing_center_y_m_);
  }

  bool pathIntersectsCrossing(const RobotRuntime & runtime) const
  {
    if (distanceToCrossing(runtime.pose) <= crossing_entry_radius_m_) {
      return true;
    }

    DirectMppiState previous = runtime.pose;
    if (runtime.offline_path.empty()) {
      return distanceToCrossingSegment(previous, runtime.goal) <= crossing_zone_radius_m_;
    }

    const size_t start_index =
      std::min(runtime.waypoint_index, runtime.offline_path.size() - 1);
    for (size_t i = start_index; i < runtime.offline_path.size(); ++i) {
      const DirectMppiState next = runtime.offline_path[i];
      if (distanceToCrossingSegment(previous, next) <= crossing_zone_radius_m_) {
        return true;
      }
      previous = next;
    }
    return false;
  }

  bool crossingCandidate(const RobotRuntime & runtime) const
  {
    if (!runtime.has_goal ||
      distance2d(runtime.pose, runtime.goal) <= planner_params_.goal_radius_m)
    {
      return false;
    }
    return pathIntersectsCrossing(runtime);
  }

  int robotIndex(const std::string & robot) const
  {
    const auto it = std::find(robot_names_.begin(), robot_names_.end(), robot);
    if (it == robot_names_.end()) {
      return -1;
    }
    return static_cast<int>(std::distance(robot_names_.begin(), it));
  }

  size_t chooseCrossingToken(const std::vector<bool> & candidates) const
  {
    const int last_index = crossing_last_token_robot_ ?
      robotIndex(*crossing_last_token_robot_) :
      -1;

    auto choose_after_last = [&](bool only_inside_zone) -> size_t {
        for (size_t offset = 1; offset <= robot_names_.size(); ++offset) {
          const size_t idx = static_cast<size_t>(
            (last_index + static_cast<int>(offset) + robot_names_.size()) %
            static_cast<int>(robot_names_.size()));
          if (!candidates[idx]) {
            continue;
          }
          if (only_inside_zone &&
            distanceToCrossing(robots_.at(robot_names_[idx]).pose) > crossing_zone_radius_m_)
          {
            continue;
          }
          return idx;
        }
        return robot_names_.size();
      };

    const size_t inside_choice = choose_after_last(true);
    if (inside_choice < robot_names_.size()) {
      return inside_choice;
    }
    return choose_after_last(false);
  }

  void acquireCrossingToken(const std::string & robot)
  {
    const auto & runtime = robots_.at(robot);
    crossing_token_robot_ = robot;
    crossing_last_token_robot_ = robot;
    crossing_token_entered_zone_ =
      distanceToCrossing(runtime.pose) <= crossing_zone_radius_m_;
    crossing_token_acquired_time_ = this->now();
    crossing_token_last_progress_time_ = crossing_token_acquired_time_;
    crossing_token_best_goal_distance_ = distance2d(runtime.pose, runtime.goal);
    publishStatus("crossing gate: " + robot + " has right-of-way", true);
  }

  void releaseCrossingToken(const std::string & reason)
  {
    if (crossing_token_robot_) {
      publishStatus(
        "crossing gate: released " + *crossing_token_robot_ + " (" + reason + ")",
        true);
    }
    crossing_token_robot_.reset();
    crossing_token_entered_zone_ = false;
    crossing_token_best_goal_distance_ = std::numeric_limits<double>::infinity();
  }

  void resetCrossingGate()
  {
    crossing_token_robot_.reset();
    crossing_last_token_robot_.reset();
    crossing_token_entered_zone_ = false;
    crossing_token_best_goal_distance_ = std::numeric_limits<double>::infinity();
  }

  void updateCrossingToken(const std::vector<bool> & candidates)
  {
    if (!crossing_gate_enabled_ || robot_names_.empty()) {
      resetCrossingGate();
      return;
    }

    const double zone_radius = std::max(0.01, crossing_zone_radius_m_);
    const double entry_radius = std::max(zone_radius, crossing_entry_radius_m_);
    const double release_radius = std::max(zone_radius, crossing_release_radius_m_);
    crossing_zone_radius_m_ = zone_radius;
    crossing_entry_radius_m_ = entry_radius;
    crossing_release_radius_m_ = release_radius;

    const auto now = this->now();
    if (crossing_token_robot_) {
      const int token_index = robotIndex(*crossing_token_robot_);
      if (token_index < 0 || !candidates[static_cast<size_t>(token_index)]) {
        releaseCrossingToken("no longer needs crossing");
      } else {
        const auto & runtime = robots_.at(*crossing_token_robot_);
        const double center_dist = distanceToCrossing(runtime.pose);
        if (center_dist <= crossing_zone_radius_m_) {
          crossing_token_entered_zone_ = true;
        }

        const double goal_dist = distance2d(runtime.pose, runtime.goal);
        if (goal_dist < crossing_token_best_goal_distance_ - crossing_progress_epsilon_m_) {
          crossing_token_best_goal_distance_ = goal_dist;
          crossing_token_last_progress_time_ = now;
        }

        const bool released_after_exit =
          crossing_token_entered_zone_ && center_dist >= crossing_release_radius_m_;
        const bool reached_goal = goal_dist <= planner_params_.goal_radius_m;
        bool timed_out = false;
        if (crossing_progress_timeout_s_ > 0.0) {
          bool someone_else_waits = false;
          for (size_t i = 0; i < candidates.size(); ++i) {
            if (candidates[i] && static_cast<int>(i) != token_index) {
              someone_else_waits = true;
              break;
            }
          }
          timed_out = someone_else_waits &&
            (now - crossing_token_last_progress_time_).seconds() > crossing_progress_timeout_s_;
        }

        if (reached_goal) {
          releaseCrossingToken("goal reached");
        } else if (released_after_exit) {
          releaseCrossingToken("exited zone");
        } else if (timed_out) {
          releaseCrossingToken("progress timeout");
        }
      }
    }

    if (!crossing_token_robot_) {
      const size_t next_token = chooseCrossingToken(candidates);
      if (next_token < robot_names_.size()) {
        acquireCrossingToken(robot_names_[next_token]);
      }
    }
  }

  bool robotIsLeavingCrossing(
    const RobotRuntime & runtime,
    const WorldVelocity & velocity) const
  {
    const double dx = static_cast<double>(runtime.pose.x) - crossing_center_x_m_;
    const double dy = static_cast<double>(runtime.pose.y) - crossing_center_y_m_;
    const double dist = std::hypot(dx, dy);
    if (dist < 1e-6) {
      return false;
    }
    const double radial_speed = (velocity.x * dx + velocity.y * dy) / dist;
    return radial_speed > 0.01;
  }

  std::vector<bool> applyCrossingGate(
    std::vector<DirectMppiControl> & preferred_controls,
    std::vector<WorldVelocity> & world_velocities)
  {
    std::vector<bool> held(robot_names_.size(), false);
    if (!crossing_gate_enabled_ || control_mode_ != "scheduled") {
      return held;
    }

    std::vector<bool> candidates(robot_names_.size(), false);
    for (size_t i = 0; i < robot_names_.size(); ++i) {
      candidates[i] = crossingCandidate(robots_.at(robot_names_[i]));
    }

    updateCrossingToken(candidates);
    if (!crossing_token_robot_) {
      return held;
    }

    for (size_t i = 0; i < robot_names_.size(); ++i) {
      if (!candidates[i] || robot_names_[i] == *crossing_token_robot_) {
        continue;
      }

      const auto & runtime = robots_.at(robot_names_[i]);
      const double center_dist = distanceToCrossing(runtime.pose);
      if (center_dist > crossing_entry_radius_m_) {
        continue;
      }
      if (robotIsLeavingCrossing(runtime, world_velocities[i])) {
        continue;
      }

      preferred_controls[i].v = 0.0f;
      world_velocities[i] = WorldVelocity{};
      held[i] = true;
    }
    return held;
  }

  DirectMppiControl worldToControl(
    const DirectMppiState & pose,
    const WorldVelocity & velocity) const
  {
    const double speed = std::hypot(velocity.x, velocity.y);
    DirectMppiControl control;
    if (speed < 1e-4) {
      return control;
    }

    const double desired_heading = std::atan2(velocity.y, velocity.x);
    double heading_error = normalizeAngle(desired_heading - pose.theta);
    bool reverse = false;
    if (allow_reverse_ && std::abs(heading_error) > reverse_heading_threshold_rad_) {
      reverse = true;
      heading_error = normalizeAngle(desired_heading + M_PI - pose.theta);
    }

    const double speed_limit = reverse ?
      std::max(0.0, max_reverse_v_mps_) :
      static_cast<double>(planner_params_.max_v_mps);
    double signed_speed = std::min(speed, speed_limit);
    if (std::abs(heading_error) > path_slow_heading_rad_) {
      signed_speed *= 0.35;
    }
    if (std::abs(heading_error) > path_stop_heading_rad_) {
      signed_speed = 0.0;
    }
    if (reverse) {
      signed_speed = -signed_speed;
    }

    control.v = static_cast<float>(signed_speed);
    control.omega = static_cast<float>(std::clamp(
      path_heading_gain_ * heading_error,
      -static_cast<double>(planner_params_.max_w_radps),
      static_cast<double>(planner_params_.max_w_radps)));
    return control;
  }

  DirectMppiControl mergeFilteredLinearWithPreferredTurn(
    const DirectMppiState & pose,
    const DirectMppiControl & preferred,
    const WorldVelocity & filtered_velocity) const
  {
    DirectMppiControl control = preferred;
    if (std::abs(static_cast<double>(preferred.v)) < 1e-4) {
      control.v = 0.0f;
      return control;
    }

    const double sign = preferred.v < 0.0f ? -1.0 : 1.0;
    const double forward_x = std::cos(static_cast<double>(pose.theta)) * sign;
    const double forward_y = std::sin(static_cast<double>(pose.theta)) * sign;
    const double projected_speed =
      filtered_velocity.x * forward_x + filtered_velocity.y * forward_y;
    const double preferred_abs_speed = std::abs(static_cast<double>(preferred.v));
    const double filtered_abs_speed = std::clamp(projected_speed, 0.0, preferred_abs_speed);
    control.v = static_cast<float>(sign * filtered_abs_speed);
    control.omega = preferred.omega;
    return control;
  }

  void clampWorldVelocity(WorldVelocity & velocity) const
  {
    const double speed = std::hypot(velocity.x, velocity.y);
    const double max_speed = static_cast<double>(planner_params_.max_v_mps);
    if (speed > max_speed && speed > 1e-6) {
      velocity.x *= max_speed / speed;
      velocity.y *= max_speed / speed;
    }
  }

  double controlStepSeconds() const
  {
    return 1.0 / std::max(1.0, control_rate_hz_);
  }

  void updateOrcaPriorityAging(
    const std::vector<DirectMppiControl> & preferred_controls,
    const std::vector<bool> & crossing_gate_holds)
  {
    if (!orca_priority_enabled_) {
      for (auto & entry : robots_) {
        entry.second.orca_priority_wait_s = 0.0;
      }
      return;
    }

    const double dt = controlStepSeconds();
    const double decay = std::max(0.0, orca_priority_wait_decay_gain_) * dt;
    for (size_t i = 0; i < robot_names_.size(); ++i) {
      auto & runtime = robots_.at(robot_names_[i]);
      const bool wants_motion =
        i < preferred_controls.size() &&
        std::abs(static_cast<double>(preferred_controls[i].v)) > 0.01;
      const bool moving_slowly =
        std::abs(static_cast<double>(runtime.measured_cv_speed_mps)) <
        std::max(0.0, orca_priority_wait_speed_mps_);
      const bool held_by_gate =
        crossing_gate_holds.size() == robot_names_.size() && crossing_gate_holds[i];

      if (held_by_gate || (wants_motion && moving_slowly)) {
        runtime.orca_priority_wait_s += dt;
      } else {
        runtime.orca_priority_wait_s = std::max(0.0, runtime.orca_priority_wait_s - decay);
      }
    }
  }

  std::vector<double> computeOrcaPriorityScores() const
  {
    std::vector<double> priorities(robot_names_.size(), 0.0);
    if (!orca_priority_enabled_) {
      return priorities;
    }

    const int schedule_index = currentScheduleIndex();
    const double fixed_bias = std::max(0.0, orca_priority_fixed_bias_);
    const double wait_gain = std::max(0.0, orca_priority_wait_gain_);
    const double lag_gain = std::max(0.0, orca_priority_schedule_lag_gain_);
    for (size_t i = 0; i < robot_names_.size(); ++i) {
      const auto & runtime = robots_.at(robot_names_[i]);
      const double deterministic_bias =
        fixed_bias * static_cast<double>(robot_names_.size() - 1 - i);
      const double schedule_lag =
        runtime.offline_path.empty() ?
        0.0 :
        std::max(0.0, static_cast<double>(schedule_index) -
        static_cast<double>(runtime.waypoint_index));
      priorities[i] =
        deterministic_bias +
        wait_gain * runtime.orca_priority_wait_s +
        lag_gain * schedule_lag;
    }
    return priorities;
  }

  std::pair<double, double> priorityAvoidanceShares(
    size_t i,
    size_t j,
    const std::vector<bool> & crossing_gate_holds,
    const std::vector<double> & orca_priorities) const
  {
    double share_i = 0.5;
    double share_j = 0.5;

    if (orca_priority_enabled_ && orca_priorities.size() == robot_names_.size()) {
      const double priority_diff = orca_priorities[j] - orca_priorities[i];
      const double bias = std::tanh(std::max(0.0, orca_priority_strength_) * priority_diff);
      share_i = 0.5 + 0.5 * bias;
      const double min_share = std::clamp(orca_priority_min_share_, 0.0, 0.49);
      share_i = std::clamp(share_i, min_share, 1.0 - min_share);
      share_j = 1.0 - share_i;
    }

    if (crossing_gate_holds.size() == robot_names_.size()) {
      if (crossing_gate_holds[i] && !crossing_gate_holds[j]) {
        share_i = 1.0;
        share_j = 0.0;
      } else if (!crossing_gate_holds[i] && crossing_gate_holds[j]) {
        share_i = 0.0;
        share_j = 1.0;
      }
    }
    return {share_i, share_j};
  }

  void applyOrcaLikeFilter(
    std::vector<WorldVelocity> & velocities,
    const std::vector<bool> & crossing_gate_holds,
    const std::vector<double> & orca_priorities) const
  {
    const double min_dist = std::max(
      static_cast<double>(planner_params_.safety_distance_m),
      2.0 * orca_radius_m_);
    const double horizon = std::max(0.1, orca_time_horizon_s_);

    for (size_t i = 0; i < robot_names_.size(); ++i) {
      for (size_t j = i + 1; j < robot_names_.size(); ++j) {
        const auto & a = robots_.at(robot_names_[i]).pose;
        const auto & b = robots_.at(robot_names_[j]).pose;
        const double dx = static_cast<double>(a.x - b.x);
        const double dy = static_cast<double>(a.y - b.y);
        const double dist = std::hypot(dx, dy);
        if (dist > orca_neighbor_dist_m_) {
          continue;
        }

        const double ux = dist > 1e-6 ? dx / dist : 1.0;
        const double uy = dist > 1e-6 ? dy / dist : 0.0;
        const double rvx = velocities[i].x - velocities[j].x;
        const double rvy = velocities[i].y - velocities[j].y;
        const double closing_speed = -(rvx * ux + rvy * uy);
        const double predicted_dist = dist - std::max(0.0, closing_speed) * horizon;
        if (predicted_dist >= min_dist && dist >= min_dist) {
          continue;
        }

        const double overlap_term = std::max(0.0, min_dist - dist) * 2.0;
        const double horizon_term = std::max(0.0, min_dist - predicted_dist) / horizon;
        const auto shares = priorityAvoidanceShares(
          i, j, crossing_gate_holds, orca_priorities);
        const double share_i = shares.first;
        const double share_j = shares.second;

        const double correction = orca_avoidance_gain_ * (overlap_term + horizon_term);
        velocities[i].x += share_i * correction * ux;
        velocities[i].y += share_i * correction * uy;
        velocities[j].x -= share_j * correction * ux;
        velocities[j].y -= share_j * correction * uy;
      }
    }

    for (auto & velocity : velocities) {
      clampWorldVelocity(velocity);
    }
  }

  void applyBoundaryVelocityFilter(std::vector<WorldVelocity> & velocities) const
  {
    const double half = safeHalfRoom();
    const double margin = std::max(1e-3, boundary_slowdown_margin_m_);
    for (size_t i = 0; i < robot_names_.size(); ++i) {
      const auto & pose = robots_.at(robot_names_[i]).pose;
      if (pose.x > half - margin && velocities[i].x > 0.0) {
        velocities[i].x *= std::clamp((half - pose.x) / margin, 0.0, 1.0);
      }
      if (pose.x < -half + margin && velocities[i].x < 0.0) {
        velocities[i].x *= std::clamp((pose.x + half) / margin, 0.0, 1.0);
      }
      if (pose.y > half - margin && velocities[i].y > 0.0) {
        velocities[i].y *= std::clamp((half - pose.y) / margin, 0.0, 1.0);
      }
      if (pose.y < -half + margin && velocities[i].y < 0.0) {
        velocities[i].y *= std::clamp((pose.y + half) / margin, 0.0, 1.0);
      }
      clampWorldVelocity(velocities[i]);
    }
  }

  void publishScheduledPreviewPaths()
  {
    for (const auto & robot : robot_names_) {
      auto & runtime = robots_.at(robot);
      if (runtime.offline_path.empty()) {
        publishPath(robot, {});
        continue;
      }
      const size_t target_index = scheduledTargetIndex(runtime);
      std::vector<DirectMppiState> remaining;
      remaining.push_back(runtime.pose);
      for (size_t i = target_index; i < runtime.offline_path.size(); ++i) {
        remaining.push_back(runtime.offline_path[i]);
      }
      publishPath(robot, remaining);
    }
  }

  bool allGoalsReady() const
  {
    for (const auto & robot : robot_names_) {
      const auto it = robots_.find(robot);
      if (it == robots_.end() || !it->second.has_goal) {
        return false;
      }
    }
    return true;
  }

  bool readyToPlan(std::string & reason) const
  {
    const auto now = this->now();
    for (const auto & robot : robot_names_) {
      const auto it = robots_.find(robot);
      if (it == robots_.end() || !it->second.has_goal) {
        reason = "missing goal for " + robot;
        return false;
      }
      if (!it->second.has_pose || !it->second.pose_time) {
        reason = "missing CV pose for " + robot;
        return false;
      }
      const double age = (now - *it->second.pose_time).seconds();
      if (age > pose_timeout_s_) {
        reason = "stale CV pose for " + robot;
        return false;
      }
      if (!insideSafeBoundary(it->second.pose, reason)) {
        reason = robot + " pose " + reason;
        return false;
      }
      if (!insideSafeBoundary(it->second.goal, reason)) {
        reason = robot + " goal " + reason;
        return false;
      }
    }
    return true;
  }

  bool readyToRun(std::string & reason) const
  {
    if (!readyToPlan(reason)) {
      return false;
    }
    return liveSpacingSafe(reason);
  }

  bool liveSpacingSafe(std::string & reason) const
  {
    for (size_t i = 0; i < robot_names_.size(); ++i) {
      for (size_t j = i + 1; j < robot_names_.size(); ++j) {
        const auto & a = robots_.at(robot_names_[i]).pose;
        const auto & b = robots_.at(robot_names_[j]).pose;
        const double d = distance2d(a, b);
        if (d < min_live_spacing_m_) {
          std::ostringstream stream;
          stream << "unsafe live spacing between " << robot_names_[i] << " and "
                 << robot_names_[j] << ": " << d << " m";
          reason = stream.str();
          return false;
        }
      }
    }
    return true;
  }

  bool offlinePathsMissing() const
  {
    if (!offline_planner_enabled_) {
      return false;
    }
    for (const auto & robot : robot_names_) {
      const auto it = robots_.find(robot);
      if (it == robots_.end() || it->second.offline_path.empty()) {
        return true;
      }
    }
    return false;
  }

  bool finalGoalsReached() const
  {
    for (const auto & robot : robot_names_) {
      const auto & runtime = robots_.at(robot);
      if (distance2d(runtime.pose, runtime.goal) > planner_params_.goal_radius_m) {
        return false;
      }
    }
    return true;
  }

  struct GridCell
  {
    int x{0};
    int y{0};
  };

  static int64_t cellTimeKey(GridCell cell, int time_index)
  {
    return (static_cast<int64_t>(time_index) << 40) ^
      (static_cast<int64_t>(cell.x) << 20) ^
      static_cast<int64_t>(cell.y);
  }

  double safeHalfRoom() const
  {
    return std::max(
      0.0,
      0.5 * static_cast<double>(planner_params_.room_size_m) -
      static_cast<double>(planner_params_.wall_margin_m));
  }

  int gridSize() const
  {
    return static_cast<int>(std::floor((2.0 * safeHalfRoom()) / offline_grid_resolution_m_)) + 1;
  }

  GridCell worldToGrid(const DirectMppiState & state) const
  {
    const double half = safeHalfRoom();
    const int size = gridSize();
    GridCell cell;
    cell.x = static_cast<int>(std::llround((state.x + half) / offline_grid_resolution_m_));
    cell.y = static_cast<int>(std::llround((state.y + half) / offline_grid_resolution_m_));
    cell.x = std::clamp(cell.x, 0, size - 1);
    cell.y = std::clamp(cell.y, 0, size - 1);
    return cell;
  }

  DirectMppiState gridToWorld(GridCell cell) const
  {
    const double half = safeHalfRoom();
    DirectMppiState state;
    state.x = static_cast<float>(-half + static_cast<double>(cell.x) * offline_grid_resolution_m_);
    state.y = static_cast<float>(-half + static_cast<double>(cell.y) * offline_grid_resolution_m_);
    state.theta = 0.0f;
    return state;
  }

  static bool sameCell(GridCell a, GridCell b)
  {
    return a.x == b.x && a.y == b.y;
  }

  bool cellReserved(
    GridCell cell,
    int time_index,
    const std::vector<std::vector<GridCell>> & reservations) const
  {
    for (const auto & path : reservations) {
      if (path.empty()) {
        continue;
      }
      const int idx = std::min(time_index, static_cast<int>(path.size()) - 1);
      if (sameCell(cell, path[idx])) {
        return true;
      }
    }
    return false;
  }

  bool edgeSwapReserved(
    GridCell from,
    GridCell to,
    int next_time_index,
    const std::vector<std::vector<GridCell>> & reservations) const
  {
    if (next_time_index <= 0) {
      return false;
    }
    for (const auto & path : reservations) {
      if (path.empty()) {
        continue;
      }
      const int prev_idx = std::min(next_time_index - 1, static_cast<int>(path.size()) - 1);
      const int next_idx = std::min(next_time_index, static_cast<int>(path.size()) - 1);
      if (sameCell(path[prev_idx], to) && sameCell(path[next_idx], from)) {
        return true;
      }
    }
    return false;
  }

  struct CbsConstraint
  {
    size_t robot_index{0};
    GridCell cell;
    GridCell from;
    GridCell to;
    int time_index{0};
    bool edge{false};
  };

  struct CbsConflict
  {
    size_t a{0};
    size_t b{0};
    GridCell cell;
    GridCell a_from;
    GridCell a_to;
    GridCell b_from;
    GridCell b_to;
    int time_index{0};
    bool edge{false};
  };

  struct CbsNode
  {
    std::vector<CbsConstraint> constraints;
    std::vector<std::vector<GridCell>> paths;
    double cost{0.0};
    int conflicts{0};
    int id{0};
  };

  struct CbsNodeCompare
  {
    bool operator()(const CbsNode & a, const CbsNode & b) const
    {
      if (std::abs(a.cost - b.cost) > 1e-9) {
        return a.cost > b.cost;
      }
      if (a.conflicts != b.conflicts) {
        return a.conflicts > b.conflicts;
      }
      return a.id > b.id;
    }
  };

  static GridCell pathCellAt(const std::vector<GridCell> & path, int time_index)
  {
    if (path.empty()) {
      return GridCell{};
    }
    const int idx = std::clamp(time_index, 0, static_cast<int>(path.size()) - 1);
    return path[idx];
  }

  bool violatesConstraint(
    size_t robot_index,
    GridCell from,
    GridCell to,
    int next_time_index,
    const std::vector<CbsConstraint> & constraints) const
  {
    for (const auto & constraint : constraints) {
      if (constraint.robot_index != robot_index || constraint.time_index != next_time_index) {
        continue;
      }
      if (constraint.edge) {
        if (sameCell(constraint.from, from) && sameCell(constraint.to, to)) {
          return true;
        }
      } else if (sameCell(constraint.cell, to)) {
        return true;
      }
    }
    return false;
  }

  int latestGoalVertexConstraintTime(
    size_t robot_index,
    GridCell goal,
    const std::vector<CbsConstraint> & constraints) const
  {
    int latest = -1;
    for (const auto & constraint : constraints) {
      if (constraint.robot_index == robot_index && !constraint.edge &&
        sameCell(constraint.cell, goal))
      {
        latest = std::max(latest, constraint.time_index);
      }
    }
    return latest;
  }

  bool planOneConstrainedTemporalAstar(
    size_t robot_index,
    const DirectMppiState & start,
    const DirectMppiState & goal,
    const std::vector<CbsConstraint> & constraints,
    std::vector<GridCell> & path,
    std::string & reason) const
  {
    const int size = gridSize();
    if (size <= 1) {
      reason = "offline grid is too small";
      return false;
    }

    const GridCell start_cell = worldToGrid(start);
    const GridCell goal_cell = worldToGrid(goal);
    if (violatesConstraint(robot_index, start_cell, start_cell, 0, constraints)) {
      reason = "start cell violates a CBS constraint";
      return false;
    }

    struct SearchNode
    {
      GridCell cell;
      int t;
      double g;
      double f;
    };
    struct Compare
    {
      bool operator()(const SearchNode & a, const SearchNode & b) const
      {
        return a.f > b.f;
      }
    };

    auto heuristic = [&](GridCell cell) {
        const double dx = static_cast<double>(cell.x - goal_cell.x);
        const double dy = static_cast<double>(cell.y - goal_cell.y);
        return std::hypot(dx, dy) * offline_grid_resolution_m_;
      };

    std::priority_queue<SearchNode, std::vector<SearchNode>, Compare> open;
    std::unordered_map<int64_t, double> best_g;
    std::unordered_map<int64_t, int64_t> parent;
    const int64_t start_key = cellTimeKey(start_cell, 0);
    open.push(SearchNode{start_cell, 0, 0.0, heuristic(start_cell)});
    best_g[start_key] = 0.0;

    const std::vector<GridCell> moves = {
      {0, 0},
      {1, 0}, {-1, 0}, {0, 1}, {0, -1},
      {1, 1}, {1, -1}, {-1, 1}, {-1, -1},
    };
    const int latest_goal_constraint =
      latestGoalVertexConstraintTime(robot_index, goal_cell, constraints);

    int64_t goal_key = 0;
    bool found = false;
    while (!open.empty()) {
      const SearchNode current = open.top();
      open.pop();
      const int64_t current_key = cellTimeKey(current.cell, current.t);
      const auto best_it = best_g.find(current_key);
      if (best_it != best_g.end() && current.g > best_it->second + 1e-9) {
        continue;
      }

      if (sameCell(current.cell, goal_cell) && current.t > latest_goal_constraint) {
        goal_key = current_key;
        found = true;
        break;
      }
      if (current.t >= offline_max_time_steps_) {
        continue;
      }

      for (const auto & move : moves) {
        GridCell next{current.cell.x + move.x, current.cell.y + move.y};
        if (next.x < 0 || next.x >= size || next.y < 0 || next.y >= size) {
          continue;
        }
        const int next_t = current.t + 1;
        if (violatesConstraint(robot_index, current.cell, next, next_t, constraints)) {
          continue;
        }
        const double move_cost = (move.x == 0 && move.y == 0) ?
          std::max(offline_wait_penalty_, 1e-3) :
          std::hypot(static_cast<double>(move.x), static_cast<double>(move.y)) *
          offline_grid_resolution_m_;
        const double next_g = current.g + move_cost;
        const int64_t next_key = cellTimeKey(next, next_t);
        const auto previous = best_g.find(next_key);
        if (previous != best_g.end() && previous->second <= next_g) {
          continue;
        }
        best_g[next_key] = next_g;
        parent[next_key] = current_key;
        open.push(SearchNode{next, next_t, next_g, next_g + heuristic(next)});
      }
    }

    if (!found) {
      reason = "no CBS-constrained temporal path found";
      return false;
    }

    std::vector<GridCell> reversed;
    int64_t key = goal_key;
    while (true) {
      const int t = static_cast<int>(key >> 40);
      const int x = static_cast<int>((key >> 20) & ((1 << 20) - 1));
      const int y = static_cast<int>(key & ((1 << 20) - 1));
      reversed.push_back(GridCell{x, y});
      if (t == 0) {
        break;
      }
      key = parent[key];
    }
    path.assign(reversed.rbegin(), reversed.rend());
    return true;
  }

  int countPathConflicts(const std::vector<std::vector<GridCell>> & paths) const
  {
    CbsConflict conflict;
    int conflicts = 0;
    const int horizon = std::min(offline_max_time_steps_, maxPathLength(paths) + 10);
    for (int t = 0; t <= horizon; ++t) {
      for (size_t a = 0; a < paths.size(); ++a) {
        for (size_t b = a + 1; b < paths.size(); ++b) {
          if (conflictAt(paths, a, b, t, conflict)) {
            ++conflicts;
          }
        }
      }
    }
    return conflicts;
  }

  int maxPathLength(const std::vector<std::vector<GridCell>> & paths) const
  {
    int max_len = 0;
    for (const auto & path : paths) {
      max_len = std::max(max_len, static_cast<int>(path.size()));
    }
    return max_len;
  }

  bool conflictAt(
    const std::vector<std::vector<GridCell>> & paths,
    size_t a,
    size_t b,
    int t,
    CbsConflict & conflict) const
  {
    const GridCell a_now = pathCellAt(paths[a], t);
    const GridCell b_now = pathCellAt(paths[b], t);
    if (sameCell(a_now, b_now)) {
      conflict.a = a;
      conflict.b = b;
      conflict.cell = a_now;
      conflict.time_index = t;
      conflict.edge = false;
      return true;
    }
    if (t <= 0) {
      return false;
    }

    const GridCell a_prev = pathCellAt(paths[a], t - 1);
    const GridCell b_prev = pathCellAt(paths[b], t - 1);
    if (sameCell(a_prev, b_now) && sameCell(a_now, b_prev)) {
      conflict.a = a;
      conflict.b = b;
      conflict.a_from = a_prev;
      conflict.a_to = a_now;
      conflict.b_from = b_prev;
      conflict.b_to = b_now;
      conflict.time_index = t;
      conflict.edge = true;
      return true;
    }
    return false;
  }

  bool findFirstConflict(
    const std::vector<std::vector<GridCell>> & paths,
    CbsConflict & conflict) const
  {
    const int horizon = std::min(offline_max_time_steps_, maxPathLength(paths) + 10);
    for (int t = 0; t <= horizon; ++t) {
      for (size_t a = 0; a < paths.size(); ++a) {
        for (size_t b = a + 1; b < paths.size(); ++b) {
          if (conflictAt(paths, a, b, t, conflict)) {
            return true;
          }
        }
      }
    }
    return false;
  }

  double pathCost(const std::vector<std::vector<GridCell>> & paths) const
  {
    double cost = 0.0;
    for (const auto & path : paths) {
      cost += static_cast<double>(path.size());
    }
    return cost;
  }

  bool planOfflinePathsCbs(std::string & reason)
  {
    CbsNode root;
    root.paths.resize(robot_names_.size());
    for (size_t i = 0; i < robot_names_.size(); ++i) {
      const auto & runtime = robots_.at(robot_names_[i]);
      std::string robot_reason;
      if (!planOneConstrainedTemporalAstar(
          i, runtime.pose, runtime.goal, root.constraints, root.paths[i], robot_reason))
      {
        reason = robot_names_[i] + ": " + robot_reason;
        return false;
      }
    }
    root.cost = pathCost(root.paths);
    root.conflicts = countPathConflicts(root.paths);

    int next_id = 1;
    root.id = next_id++;
    std::priority_queue<CbsNode, std::vector<CbsNode>, CbsNodeCompare> open;
    open.push(root);

    int expanded = 0;
    while (!open.empty() && expanded < cbs_max_nodes_) {
      CbsNode node = open.top();
      open.pop();
      ++expanded;

      CbsConflict conflict;
      if (!findFirstConflict(node.paths, conflict)) {
        applyGridPlans(node.paths);
        std::ostringstream stream;
        stream << "CBS scheduled paths ready after " << expanded << " node(s)";
        publishStatus(stream.str(), true);
        return true;
      }

      for (size_t branch = 0; branch < 2; ++branch) {
        const size_t constrained_robot = branch == 0 ? conflict.a : conflict.b;
        CbsNode child = node;
        child.id = next_id++;

        CbsConstraint constraint;
        constraint.robot_index = constrained_robot;
        constraint.time_index = conflict.time_index;
        if (conflict.edge) {
          constraint.edge = true;
          if (constrained_robot == conflict.a) {
            constraint.from = conflict.a_from;
            constraint.to = conflict.a_to;
          } else {
            constraint.from = conflict.b_from;
            constraint.to = conflict.b_to;
          }
        } else {
          constraint.edge = false;
          constraint.cell = conflict.cell;
        }
        child.constraints.push_back(constraint);

        const auto & runtime = robots_.at(robot_names_[constrained_robot]);
        std::string robot_reason;
        if (!planOneConstrainedTemporalAstar(
            constrained_robot,
            runtime.pose,
            runtime.goal,
            child.constraints,
            child.paths[constrained_robot],
            robot_reason))
        {
          continue;
        }
        child.cost = pathCost(child.paths);
        child.conflicts = countPathConflicts(child.paths);
        open.push(std::move(child));
      }
    }

    std::ostringstream stream;
    stream << "CBS failed to find a conflict-free schedule within " << cbs_max_nodes_
           << " node(s)";
    reason = stream.str();
    return false;
  }

  bool planOneTemporalAstar(
    const DirectMppiState & start,
    const DirectMppiState & goal,
    const std::vector<std::vector<GridCell>> & reservations,
    std::vector<GridCell> & path,
    std::string & reason) const
  {
    const int size = gridSize();
    if (size <= 1) {
      reason = "offline grid is too small";
      return false;
    }

    const GridCell start_cell = worldToGrid(start);
    const GridCell goal_cell = worldToGrid(goal);
    if (cellReserved(start_cell, 0, reservations)) {
      reason = "start cell is reserved by another robot";
      return false;
    }

    struct SearchNode
    {
      GridCell cell;
      int t;
      double g;
      double f;
    };
    struct Compare
    {
      bool operator()(const SearchNode & a, const SearchNode & b) const
      {
        return a.f > b.f;
      }
    };

    auto heuristic = [&](GridCell cell) {
        const double dx = static_cast<double>(cell.x - goal_cell.x);
        const double dy = static_cast<double>(cell.y - goal_cell.y);
        return std::hypot(dx, dy) * offline_grid_resolution_m_;
      };

    std::priority_queue<SearchNode, std::vector<SearchNode>, Compare> open;
    std::unordered_map<int64_t, double> best_g;
    std::unordered_map<int64_t, int64_t> parent;
    const int64_t start_key = cellTimeKey(start_cell, 0);
    open.push(SearchNode{start_cell, 0, 0.0, heuristic(start_cell)});
    best_g[start_key] = 0.0;

    const std::vector<GridCell> moves = {
      {0, 0},
      {1, 0}, {-1, 0}, {0, 1}, {0, -1},
      {1, 1}, {1, -1}, {-1, 1}, {-1, -1},
    };

    int64_t goal_key = 0;
    bool found = false;
    while (!open.empty()) {
      const SearchNode current = open.top();
      open.pop();
      const int64_t current_key = cellTimeKey(current.cell, current.t);
      const auto best_it = best_g.find(current_key);
      if (best_it != best_g.end() && current.g > best_it->second + 1e-9) {
        continue;
      }

      if (sameCell(current.cell, goal_cell)) {
        goal_key = current_key;
        found = true;
        break;
      }
      if (current.t >= offline_max_time_steps_) {
        continue;
      }

      for (const auto & move : moves) {
        GridCell next{current.cell.x + move.x, current.cell.y + move.y};
        if (next.x < 0 || next.x >= size || next.y < 0 || next.y >= size) {
          continue;
        }
        const int next_t = current.t + 1;
        if (cellReserved(next, next_t, reservations) ||
          edgeSwapReserved(current.cell, next, next_t, reservations))
        {
          continue;
        }
        const double move_cost = (move.x == 0 && move.y == 0) ?
          offline_wait_penalty_ :
          std::hypot(static_cast<double>(move.x), static_cast<double>(move.y)) *
          offline_grid_resolution_m_;
        const double next_g = current.g + move_cost;
        const int64_t next_key = cellTimeKey(next, next_t);
        const auto previous = best_g.find(next_key);
        if (previous != best_g.end() && previous->second <= next_g) {
          continue;
        }
        best_g[next_key] = next_g;
        parent[next_key] = current_key;
        open.push(SearchNode{next, next_t, next_g, next_g + heuristic(next)});
      }
    }

    if (!found) {
      reason = "no temporal A* path found";
      return false;
    }

    std::vector<GridCell> reversed;
    int64_t key = goal_key;
    while (true) {
      const int t = static_cast<int>(key >> 40);
      const int x = static_cast<int>((key >> 20) & ((1 << 20) - 1));
      const int y = static_cast<int>(key & ((1 << 20) - 1));
      reversed.push_back(GridCell{x, y});
      if (t == 0) {
        break;
      }
      key = parent[key];
    }
    path.assign(reversed.rbegin(), reversed.rend());
    return true;
  }

  void applyGridPlans(const std::vector<std::vector<GridCell>> & grid_paths)
  {
    for (size_t i = 0; i < robot_names_.size(); ++i) {
      const auto & robot = robot_names_[i];
      auto & runtime = robots_.at(robot);
      runtime.offline_path.clear();
      runtime.offline_path.reserve(grid_paths[i].size());
      for (const auto & cell : grid_paths[i]) {
        runtime.offline_path.push_back(gridToWorld(cell));
      }
      if (!runtime.offline_path.empty()) {
        runtime.offline_path.front() = runtime.pose;
        runtime.offline_path.back() = runtime.goal;
      }
      runtime.waypoint_index = 0;
      publishOfflinePath(robot, runtime.offline_path);
    }
  }

  bool planOfflinePathsPrioritized(std::string & reason)
  {
    std::vector<std::vector<GridCell>> reservations;
    std::vector<std::vector<GridCell>> grid_paths;
    grid_paths.reserve(robot_names_.size());
    for (const auto & robot : robot_names_) {
      auto & runtime = robots_.at(robot);
      std::vector<GridCell> grid_path;
      std::string robot_reason;
      if (!planOneTemporalAstar(runtime.pose, runtime.goal, reservations, grid_path, robot_reason)) {
        reason = robot + ": " + robot_reason;
        return false;
      }

      reservations.push_back(grid_path);
      grid_paths.push_back(grid_path);
    }
    applyGridPlans(grid_paths);
    publishStatus("prioritized temporal A* scheduled paths ready", true);
    return true;
  }

  bool planOfflinePaths(std::string & reason)
  {
    bool success = false;
    if (planner_algorithm_ == "cbs") {
      success = planOfflinePathsCbs(reason);
    } else if (planner_algorithm_ == "prioritized") {
      success = planOfflinePathsPrioritized(reason);
    } else {
      reason = "unknown planner_algorithm '" + planner_algorithm_ + "'";
      return false;
    }
    if (!success) {
      return false;
    }
    plan_requested_ = false;
    return true;
  }

  DirectMppiState selectLocalGoal(RobotRuntime & runtime)
  {
    if (!offline_planner_enabled_ || runtime.offline_path.empty()) {
      return runtime.goal;
    }

    while (runtime.waypoint_index + 1 < runtime.offline_path.size() &&
      distance2d(runtime.pose, runtime.offline_path[runtime.waypoint_index]) < waypoint_reached_m_)
    {
      ++runtime.waypoint_index;
    }

    size_t target_index = runtime.waypoint_index;
    double accumulated = 0.0;
    DirectMppiState previous = runtime.pose;
    while (target_index + 1 < runtime.offline_path.size() && accumulated < waypoint_lookahead_m_) {
      const DirectMppiState next = runtime.offline_path[target_index + 1];
      accumulated += distance2d(previous, next);
      previous = next;
      ++target_index;
    }
    return runtime.offline_path[std::min(target_index, runtime.offline_path.size() - 1)];
  }

  bool insideSafeBoundary(const DirectMppiState & state, std::string & reason) const
  {
    const double half = 0.5 * static_cast<double>(planner_params_.room_size_m);
    const double safe_half = std::max(
      0.0,
      half - static_cast<double>(planner_params_.wall_margin_m));
    if (std::abs(static_cast<double>(state.x)) <= safe_half &&
      std::abs(static_cast<double>(state.y)) <= safe_half)
    {
      return true;
    }

    std::ostringstream stream;
    stream << "outside safe boundary x/y=[" << -safe_half << ", " << safe_half
           << "] m with " << planner_params_.wall_margin_m << " m boundary margin";
    reason = stream.str();
    return false;
  }

  DirectMppiControl publishControl(RobotRuntime & runtime, const DirectMppiControl & control)
  {
    DirectMppiControl bounded;
    const float min_v = allow_reverse_ ?
      -static_cast<float>(std::max(0.0, max_reverse_v_mps_)) :
      0.0f;
    bounded.v = std::clamp(
      control.v,
      min_v,
      planner_params_.max_v_mps);
    bounded.omega = std::clamp(
      control.omega,
      -planner_params_.max_w_radps,
      planner_params_.max_w_radps);
    geometry_msgs::msg::Twist msg;
    msg.linear.x = bounded.v;
    msg.angular.z = bounded.omega;
    runtime.cmd_pub->publish(msg);
    return bounded;
  }

  void publishMeasuredVelocity(const RobotRuntime & runtime)
  {
    if (!runtime.actual_velocity_pub) {
      return;
    }
    geometry_msgs::msg::Twist msg;
    msg.linear.x = runtime.measured_cv_velocity.v;
    msg.linear.y = runtime.measured_cv_lateral_mps;
    msg.linear.z = runtime.measured_cv_speed_mps;
    msg.angular.z = runtime.measured_cv_velocity.omega;
    runtime.actual_velocity_pub->publish(msg);
  }

  void publishZeroBurst()
  {
    geometry_msgs::msg::Twist zero;
    for (int repeat = 0; repeat < 5; ++repeat) {
      for (auto & entry : robots_) {
        entry.second.cmd_pub->publish(zero);
      }
    }
  }

  void stopAll(const std::string & reason)
  {
    active_ = false;
    schedule_start_time_.reset();
    resetCrossingGate();
    for (auto & entry : robots_) {
      entry.second.last_control = DirectMppiControl{0.0f, 0.0f};
      entry.second.orca_priority_wait_s = 0.0;
    }
    publishZeroBurst();
    publishStatus("stopped all robots: " + reason, true);
  }

  void publishPath(
    const std::string & robot,
    const std::vector<DirectMppiState> & path_states)
  {
    nav_msgs::msg::Path path;
    path.header.frame_id = map_frame_;
    path.header.stamp = this->now();
    for (const auto & state : path_states) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = path.header;
      pose.pose.position.x = state.x;
      pose.pose.position.y = state.y;
      pose.pose.position.z = 0.0;
      pose.pose.orientation.w = 1.0;
      path.poses.push_back(pose);
    }
    robots_[robot].path_pub->publish(path);
  }

  void publishOfflinePath(
    const std::string & robot,
    const std::vector<DirectMppiState> & path_states)
  {
    nav_msgs::msg::Path path;
    path.header.frame_id = map_frame_;
    path.header.stamp = this->now();
    for (const auto & state : path_states) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = path.header;
      pose.pose.position.x = state.x;
      pose.pose.position.y = state.y;
      pose.pose.position.z = 0.0;
      pose.pose.orientation.w = 1.0;
      path.poses.push_back(pose);
    }
    robots_[robot].offline_path_pub->publish(path);
  }

  void publishVelocityStatus()
  {
    const auto now = this->now();
    if ((now - last_velocity_status_time_).seconds() < velocity_status_period_s_) {
      return;
    }
    last_velocity_status_time_ = now;

    std::ostringstream stream;
    stream << std::fixed;
    for (size_t i = 0; i < robot_names_.size(); ++i) {
      if (i != 0) {
        stream << " | ";
      }
      const auto & robot = robot_names_[i];
      const auto & runtime = robots_.at(robot);
      stream << robot
             << " cmd=(" << std::setprecision(3) << runtime.last_control.v
             << " m/s," << runtime.last_control.omega << " rad/s)"
             << " cv_fwd=(" << runtime.measured_cv_velocity.v
             << " m/s," << runtime.measured_cv_velocity.omega << " rad/s)"
             << " cv_speed=" << runtime.measured_cv_speed_mps << " m/s";
      if (runtime.has_odom_velocity) {
        stream << " odom=(" << runtime.measured_odom_velocity.v
               << " m/s," << runtime.measured_odom_velocity.omega << " rad/s)";
      }
    }

    std_msgs::msg::String msg;
    msg.data = stream.str();
    velocity_status_pub_->publish(msg);
    RCLCPP_INFO(this->get_logger(), "%s", msg.data.c_str());
  }

  void publishStatus(const std::string & text, bool force = false)
  {
    const auto now = this->now();
    if (!force && text == last_status_ && (now - last_status_time_).seconds() < 2.0) {
      return;
    }
    last_status_ = text;
    last_status_time_ = now;

    std_msgs::msg::String msg;
    msg.data = text;
    status_pub_->publish(msg);
    RCLCPP_INFO(this->get_logger(), "%s", text.c_str());
  }

  std::vector<std::string> robot_names_;
  std::map<std::string, RobotRuntime> robots_;
  std::string map_frame_;
  double control_rate_hz_{10.0};
  double pose_timeout_s_{0.5};
  double min_live_spacing_m_{0.22};
  std::string control_mode_{"scheduled"};
  std::string planner_algorithm_{"cbs"};
  bool allow_reverse_{true};
  double max_reverse_v_mps_{0.50};
  bool offline_planner_enabled_{true};
  double offline_grid_resolution_m_{0.10};
  double offline_time_step_s_{0.50};
  int offline_max_time_steps_{180};
  double offline_wait_penalty_{0.05};
  int cbs_max_nodes_{512};
  double waypoint_reached_m_{0.14};
  double waypoint_lookahead_m_{0.30};
  double path_heading_gain_{2.0};
  double reverse_heading_threshold_rad_{2.20};
  double path_slow_heading_rad_{0.70};
  double path_stop_heading_rad_{1.40};
  double path_goal_slowdown_m_{0.35};
  bool orca_filter_enabled_{true};
  double orca_radius_m_{0.16};
  double orca_neighbor_dist_m_{0.75};
  double orca_time_horizon_s_{2.0};
  double orca_avoidance_gain_{1.0};
  bool orca_priority_enabled_{true};
  double orca_priority_strength_{2.0};
  double orca_priority_fixed_bias_{0.25};
  double orca_priority_wait_gain_{0.6};
  double orca_priority_schedule_lag_gain_{0.2};
  double orca_priority_min_share_{0.10};
  double orca_priority_wait_speed_mps_{0.02};
  double orca_priority_wait_decay_gain_{2.0};
  double boundary_slowdown_margin_m_{0.15};
  bool crossing_gate_enabled_{false};
  double crossing_center_x_m_{0.0};
  double crossing_center_y_m_{0.0};
  double crossing_zone_radius_m_{0.36};
  double crossing_entry_radius_m_{0.62};
  double crossing_release_radius_m_{0.50};
  double crossing_progress_timeout_s_{4.0};
  double crossing_progress_epsilon_m_{0.03};
  double velocity_status_period_s_{1.0};
  DirectMppiParams planner_params_;
  bool active_{false};
  bool plan_requested_{false};
  std::string last_status_;
  rclcpp::Time last_status_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_velocity_status_time_{0, 0, RCL_ROS_TIME};
  std::optional<rclcpp::Time> schedule_start_time_;
  std::optional<std::string> crossing_token_robot_;
  std::optional<std::string> crossing_last_token_robot_;
  bool crossing_token_entered_zone_{false};
  double crossing_token_best_goal_distance_{std::numeric_limits<double>::infinity()};
  rclcpp::Time crossing_token_acquired_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time crossing_token_last_progress_time_{0, 0, RCL_ROS_TIME};

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr velocity_status_pub_;
  std::vector<rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr> pose_subs_;
  std::vector<rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr> goal_subs_;
  std::vector<rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr> odom_subs_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr start_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stop_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr clear_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr plan_srv_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace multi_robot_swarm_planner

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<multi_robot_swarm_planner::MppiDirectController>();
  rclcpp::spin(node);
  node->stopForShutdown();
  rclcpp::shutdown();
  return 0;
}
