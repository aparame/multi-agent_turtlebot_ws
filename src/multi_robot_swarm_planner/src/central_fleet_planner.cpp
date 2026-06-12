#include "multi_robot_swarm_planner/swarm_mppi.hpp"

#include <action_msgs/msg/goal_status.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav2_msgs/action/follow_path.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/exceptions.h>
#include <tf2/time.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <tf2_ros/buffer.h>
#include <visualization_msgs/msg/marker_array.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace std::chrono_literals;

namespace multi_robot_swarm_planner
{
namespace
{

using FollowPath = nav2_msgs::action::FollowPath;
using GoalHandleFollowPath = rclcpp_action::ClientGoalHandle<FollowPath>;

struct RobotPlan
{
  nav_msgs::msg::Path path;
  std::optional<geometry_msgs::msg::PointStamped> goal;
};

struct Color
{
  float r;
  float g;
  float b;
};

double distance2d(double ax, double ay, double bx, double by)
{
  const double dx = ax - bx;
  const double dy = ay - by;
  return std::sqrt(dx * dx + dy * dy);
}

geometry_msgs::msg::Quaternion yawToQuaternion(double yaw)
{
  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, yaw);
  return tf2::toMsg(q);
}

std::string statusName(int8_t status)
{
  switch (status) {
    case action_msgs::msg::GoalStatus::STATUS_SUCCEEDED:
      return "succeeded";
    case action_msgs::msg::GoalStatus::STATUS_CANCELED:
      return "canceled";
    case action_msgs::msg::GoalStatus::STATUS_ABORTED:
      return "aborted";
    case action_msgs::msg::GoalStatus::STATUS_EXECUTING:
      return "executing";
    case action_msgs::msg::GoalStatus::STATUS_ACCEPTED:
      return "accepted";
    default:
      return std::to_string(status);
  }
}

std::string resultCodeName(rclcpp_action::ResultCode code)
{
  switch (code) {
    case rclcpp_action::ResultCode::SUCCEEDED:
      return "succeeded";
    case rclcpp_action::ResultCode::CANCELED:
      return "canceled";
    case rclcpp_action::ResultCode::ABORTED:
      return "aborted";
    case rclcpp_action::ResultCode::UNKNOWN:
    default:
      return "unknown";
  }
}

}  // namespace

class CentralFleetPlanner : public rclcpp::Node
{
public:
  CentralFleetPlanner()
  : Node("central_fleet_planner"),
    tf_buffer_(this->get_clock())
  {
    robot_names_ = this->declare_parameter<std::vector<std::string>>(
      "robot_names", std::vector<std::string>{"tb_1", "tb_2", "tb_3"});
    map_topic_ = this->declare_parameter<std::string>("map_topic", "/tb_1/map");
    map_frame_ = this->declare_parameter<std::string>("map_frame", "map");
    base_frame_ = this->declare_parameter<std::string>("base_frame", "base_footprint");
    fallback_base_frame_ = this->declare_parameter<std::string>("fallback_base_frame", "base_link");
    execute_after_all_goals_ = this->declare_parameter<bool>("execute_after_all_goals", true);
    replan_deviation_m_ = this->declare_parameter<double>("replan_deviation_m", 0.55);
    planner_params_.goal_radius_m = static_cast<float>(this->declare_parameter<double>("goal_radius_m", 0.25));
    planner_params_.robot_radius_m = static_cast<float>(this->declare_parameter<double>("robot_radius_m", 0.22));
    planner_params_.safety_distance_m = static_cast<float>(this->declare_parameter<double>("safety_distance_m", 0.45));
    planner_params_.planning_clearance_m = static_cast<float>(this->declare_parameter<double>("planning_clearance_m", 0.55));
    planner_params_.max_v_mps = static_cast<float>(this->declare_parameter<double>("max_v_mps", 0.20));
    planner_params_.max_w_radps = static_cast<float>(this->declare_parameter<double>("max_w_radps", 1.20));
    planner_params_.dt = static_cast<float>(this->declare_parameter<double>("dt", 0.10));
    planner_params_.horizon = this->declare_parameter<int>("horizon", 80);
    planner_params_.samples = this->declare_parameter<int>("samples", 512);
    planner_params_.mppi_iterations = this->declare_parameter<int>("mppi_iterations", 6);
    planner_params_.max_plan_steps = this->declare_parameter<int>("max_plan_steps", 900);
    planner_params_.occupancy_threshold = this->declare_parameter<int>("occupancy_threshold", 50);
    planner_params_.treat_unknown_as_obstacle =
      this->declare_parameter<bool>("treat_unknown_as_obstacle", true);
    path_min_spacing_m_ = this->declare_parameter<double>("path_min_spacing_m", 0.05);
    follow_path_server_timeout_s_ =
      this->declare_parameter<double>("follow_path_server_timeout_s", 2.0);

    robot_names_.erase(
      std::remove_if(robot_names_.begin(), robot_names_.end(), [](const std::string & name) {
        return name.empty();
      }),
      robot_names_.end());

    if (robot_names_.empty()) {
      throw std::runtime_error("central_fleet_planner needs at least one robot name");
    }

    rclcpp::QoS map_qos(1);
    map_qos.reliable().transient_local();
    map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
      map_topic_, map_qos,
      [this](nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        const bool first_map = !latest_map_;
        latest_map_ = *msg;
        if (first_map) {
          publishStatusLocked("map received from " + map_topic_);
        }
      });

    rclcpp::QoS tf_qos(100);
    tf_qos.best_effort();
    rclcpp::QoS static_tf_qos(100);
    static_tf_qos.reliable().transient_local();

    status_pub_ = this->create_publisher<std_msgs::msg::String>("/fleet_planner/status", 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/fleet_markers", 10);

    for (const auto & robot : robot_names_) {
      goal_subs_.push_back(this->create_subscription<geometry_msgs::msg::PointStamped>(
        "/" + robot + "/position_goal", 10,
        [this, robot](geometry_msgs::msg::PointStamped::SharedPtr msg) {
          handleGoal(robot, *msg);
        }));

      tf_subs_.push_back(this->create_subscription<tf2_msgs::msg::TFMessage>(
        "/" + robot + "/tf", tf_qos,
        [this](tf2_msgs::msg::TFMessage::SharedPtr msg) {
          ingestTf(*msg, false);
        }));
      tf_subs_.push_back(this->create_subscription<tf2_msgs::msg::TFMessage>(
        "/" + robot + "/tf_static", static_tf_qos,
        [this](tf2_msgs::msg::TFMessage::SharedPtr msg) {
          ingestTf(*msg, true);
        }));

      path_pubs_[robot] = this->create_publisher<nav_msgs::msg::Path>("/" + robot + "/central_plan", 10);
      follow_clients_[robot] =
        rclcpp_action::create_client<FollowPath>(this, "/" + robot + "/follow_path");
      plans_.emplace(robot, RobotPlan());
    }

    clear_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "/fleet_planner/clear_goals",
      [this](
        const std::shared_ptr<std_srvs::srv::Trigger::Request>,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        std::lock_guard<std::mutex> lock(mutex_);
        cancelActiveGoalsLocked("clear_goals service");
        for (auto & entry : plans_) {
          entry.second.goal.reset();
          entry.second.path.poses.clear();
        }
        plan_requested_ = false;
        publishDeleteMarkers();
        publishStatusLocked("cleared fleet goals");
        response->success = true;
        response->message = "cleared fleet goals";
      });

    replan_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "/fleet_planner/replan",
      [this](
        const std::shared_ptr<std_srvs::srv::Trigger::Request>,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!allGoalsReadyLocked()) {
          response->success = false;
          response->message = "not all robot goals are set";
          return;
        }
        cancelActiveGoalsLocked("replan service");
        plan_requested_ = true;
        publishStatusLocked("replan requested");
        response->success = true;
        response->message = "replan requested";
      });

    timer_ = this->create_wall_timer(500ms, [this]() { timerCallback(); });

    std::ostringstream status;
    status << "central planner ready for ";
    for (size_t i = 0; i < robot_names_.size(); ++i) {
      if (i != 0) {
        status << ", ";
      }
      status << robot_names_[i];
    }
    publishStatus(status.str());
  }

private:
  void handleGoal(const std::string & robot, const geometry_msgs::msg::PointStamped & raw_goal)
  {
    geometry_msgs::msg::PointStamped goal = raw_goal;
    if (goal.header.frame_id.empty()) {
      goal.header.frame_id = map_frame_;
    }
    if (goal.header.frame_id != map_frame_) {
      try {
        geometry_msgs::msg::PointStamped transformed;
        tf_buffer_.transform(goal, transformed, map_frame_);
        goal = transformed;
      } catch (const tf2::TransformException & ex) {
        publishStatus("ignored " + robot + " goal: could not transform to " + map_frame_ + ": " + ex.what());
        return;
      }
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (planning_ || execution_active_) {
      cancelActiveGoalsLocked("new goal for " + robot);
    }
    goal.header.stamp = this->now();
    plans_[robot].goal = goal;
    publishGoalMarkersLocked();

    std::ostringstream status;
    status << "stored " << robot << " position goal: x=" << goal.point.x
           << ", y=" << goal.point.y << ", radius=" << planner_params_.goal_radius_m;
    publishStatusLocked(status.str());

    if (allGoalsReadyLocked()) {
      plan_requested_ = true;
      publishStatusLocked("all position goals received; queued centralized plan");
    }
  }

  void ingestTf(const tf2_msgs::msg::TFMessage & msg, bool is_static)
  {
    for (const auto & transform : msg.transforms) {
      try {
        tf_buffer_.setTransform(transform, "central_fleet_planner", is_static);
      } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "Could not store TF %s -> %s: %s",
          transform.header.frame_id.c_str(), transform.child_frame_id.c_str(), ex.what());
      }
    }
  }

  void timerCallback()
  {
    bool should_plan = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      should_plan = plan_requested_ && !planning_;
      if (should_plan) {
        plan_requested_ = false;
        planning_ = true;
      }
    }

    if (should_plan) {
      planAndMaybeExecute();
      std::lock_guard<std::mutex> lock(mutex_);
      planning_ = false;
      return;
    }

    monitorExecution();
  }

  void planAndMaybeExecute()
  {
    std::vector<std::string> robots;
    nav_msgs::msg::OccupancyGrid map_msg;
    std::vector<geometry_msgs::msg::PointStamped> goals;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!latest_map_) {
        publishStatusLocked("cannot plan: map has not been received");
        return;
      }
      if (!allGoalsReadyLocked()) {
        publishStatusLocked("cannot plan: waiting for every robot goal");
        return;
      }
      robots = robot_names_;
      map_msg = *latest_map_;
      goals.reserve(robots.size());
      for (const auto & robot : robots) {
        goals.push_back(*plans_[robot].goal);
      }
    }

    std::vector<PlannerState> starts;
    starts.reserve(robots.size());
    for (const auto & robot : robots) {
      auto pose = lookupRobotState(robot);
      if (!pose) {
        publishStatus("cannot plan: missing map-frame pose for " + robot);
        return;
      }
      starts.push_back(*pose);
    }

    std::vector<PlannerState> planner_goals;
    planner_goals.reserve(goals.size());
    for (const auto & goal : goals) {
      PlannerState state;
      state.x = static_cast<float>(goal.point.x);
      state.y = static_cast<float>(goal.point.y);
      state.theta = 0.0f;
      planner_goals.push_back(state);
    }

    PlannerMap planner_map;
    planner_map.width = static_cast<int>(map_msg.info.width);
    planner_map.height = static_cast<int>(map_msg.info.height);
    planner_map.resolution = map_msg.info.resolution;
    planner_map.origin_x = static_cast<float>(map_msg.info.origin.position.x);
    planner_map.origin_y = static_cast<float>(map_msg.info.origin.position.y);
    planner_map.data = map_msg.data;

    publishStatus("planning centralized fleet paths with CUDA MPPI");
    PlanResult result = planSwarmMppi(planner_map, starts, planner_goals, planner_params_);
    if (!result.success) {
      publishStatus("centralized plan failed: " + result.message);
      return;
    }

    std::map<std::string, nav_msgs::msg::Path> paths;
    for (size_t i = 0; i < robots.size(); ++i) {
      paths[robots[i]] = buildPath(robots[i], result.paths[i]);
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (const auto & robot : robots) {
        plans_[robot].path = paths[robot];
        path_pubs_[robot]->publish(paths[robot]);
      }
      publishPlanMarkersLocked();
      publishStatusLocked("centralized plan ready: " + result.message);
    }

    if (execute_after_all_goals_) {
      sendFollowPathGoals(paths);
    }
  }

  std::optional<PlannerState> lookupRobotState(const std::string & robot)
  {
    const std::array<std::string, 2> frames = {
      robot + "/" + base_frame_,
      robot + "/" + fallback_base_frame_,
    };

    for (const auto & frame : frames) {
      try {
        auto transform = tf_buffer_.lookupTransform(
          map_frame_, frame, tf2::TimePointZero);
        PlannerState state;
        state.x = static_cast<float>(transform.transform.translation.x);
        state.y = static_cast<float>(transform.transform.translation.y);
        state.theta = static_cast<float>(tf2::getYaw(transform.transform.rotation));
        return state;
      } catch (const tf2::TransformException &) {
      }
    }

    return std::nullopt;
  }

  nav_msgs::msg::Path buildPath(
    const std::string & robot,
    const std::vector<PlannerState> & states)
  {
    nav_msgs::msg::Path path;
    path.header.frame_id = map_frame_;
    path.header.stamp = this->now();

    PlannerState last_kept;
    bool has_last = false;
    for (size_t i = 0; i < states.size(); ++i) {
      const bool is_last = i + 1 == states.size();
      if (has_last && !is_last &&
        distance2d(last_kept.x, last_kept.y, states[i].x, states[i].y) < path_min_spacing_m_)
      {
        continue;
      }

      geometry_msgs::msg::PoseStamped pose;
      pose.header = path.header;
      pose.pose.position.x = states[i].x;
      pose.pose.position.y = states[i].y;
      pose.pose.position.z = 0.0;

      double yaw = states[i].theta;
      if (i + 1 < states.size()) {
        const double dx = states[i + 1].x - states[i].x;
        const double dy = states[i + 1].y - states[i].y;
        if (std::hypot(dx, dy) > 1e-4) {
          yaw = std::atan2(dy, dx);
        }
      }
      pose.pose.orientation = yawToQuaternion(yaw);
      path.poses.push_back(pose);
      last_kept = states[i];
      has_last = true;
    }

    if (path.poses.size() < 2) {
      RCLCPP_WARN(this->get_logger(), "%s central plan has fewer than 2 poses", robot.c_str());
    }
    return path;
  }

  void sendFollowPathGoals(const std::map<std::string, nav_msgs::msg::Path> & paths)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    active_handles_.clear();
    execution_active_ = true;
    completed_results_.clear();

    for (const auto & robot : robot_names_) {
      auto client = follow_clients_[robot];
      if (!client->wait_for_action_server(std::chrono::duration<double>(follow_path_server_timeout_s_))) {
        publishStatusLocked("cannot execute: /" + robot + "/follow_path action server is unavailable");
        execution_active_ = false;
        return;
      }

      FollowPath::Goal goal;
      goal.path = paths.at(robot);
      goal.controller_id = "FollowPath";
      goal.goal_checker_id = "position_goal_checker";

      rclcpp_action::Client<FollowPath>::SendGoalOptions options;
      options.goal_response_callback =
        [this, robot](GoalHandleFollowPath::SharedPtr handle) {
          std::lock_guard<std::mutex> callback_lock(mutex_);
          if (!handle) {
            publishStatusLocked(robot + " rejected centralized FollowPath goal");
            execution_active_ = false;
            return;
          }
          active_handles_[robot] = handle;
          publishStatusLocked(robot + " accepted centralized FollowPath goal");
        };
      options.result_callback =
        [this, robot](const GoalHandleFollowPath::WrappedResult & result) {
          std::lock_guard<std::mutex> callback_lock(mutex_);
          completed_results_[robot] = result.code;
          publishStatusLocked(robot + " FollowPath finished: " + resultCodeName(result.code));
          if (completed_results_.size() == robot_names_.size()) {
            execution_active_ = false;
            active_handles_.clear();
            publishStatusLocked("centralized fleet execution finished");
          }
        };

      client->async_send_goal(goal, options);
    }

    publishStatusLocked("sent all centralized FollowPath goals");
  }

  void monitorExecution()
  {
    bool should_replan = false;
    std::string reason;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!execution_active_ || planning_) {
        return;
      }
    }

    std::map<std::string, PlannerState> states;
    for (const auto & robot : robot_names_) {
      auto state = lookupRobotState(robot);
      if (!state) {
        continue;
      }
      states[robot] = *state;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (size_t i = 0; i < robot_names_.size(); ++i) {
        for (size_t j = i + 1; j < robot_names_.size(); ++j) {
          const auto & a = robot_names_[i];
          const auto & b = robot_names_[j];
          if (!states.count(a) || !states.count(b)) {
            continue;
          }
          const double d = distance2d(states[a].x, states[a].y, states[b].x, states[b].y);
          if (d < planner_params_.safety_distance_m) {
            should_replan = true;
            reason = "unsafe live robot spacing detected";
          }
        }
      }

      for (const auto & robot : robot_names_) {
        if (should_replan || !states.count(robot) || plans_[robot].path.poses.empty()) {
          continue;
        }
        const double deviation = distanceToPath(states[robot], plans_[robot].path);
        if (deviation > replan_deviation_m_) {
          should_replan = true;
          reason = robot + " deviated from centralized path";
        }
      }

      if (should_replan && allGoalsReadyLocked()) {
        cancelActiveGoalsLocked(reason);
        plan_requested_ = true;
        publishStatusLocked(reason + "; queued replan");
      }
    }
  }

  double distanceToPath(const PlannerState & state, const nav_msgs::msg::Path & path)
  {
    double best = std::numeric_limits<double>::infinity();
    for (const auto & pose : path.poses) {
      best = std::min(
        best,
        distance2d(state.x, state.y, pose.pose.position.x, pose.pose.position.y));
    }
    return best;
  }

  bool allGoalsReadyLocked() const
  {
    for (const auto & robot : robot_names_) {
      const auto it = plans_.find(robot);
      if (it == plans_.end() || !it->second.goal) {
        return false;
      }
    }
    return true;
  }

  void cancelActiveGoalsLocked(const std::string & reason)
  {
    for (const auto & entry : active_handles_) {
      const auto client_it = follow_clients_.find(entry.first);
      if (client_it != follow_clients_.end() && entry.second) {
        client_it->second->async_cancel_goal(entry.second);
      }
    }
    active_handles_.clear();
    completed_results_.clear();
    execution_active_ = false;
    publishStatusLocked("canceled active centralized paths: " + reason);
  }

  void publishGoalMarkersLocked()
  {
    visualization_msgs::msg::MarkerArray markers;
    addDeleteAll(markers);
    int id = 1;
    for (size_t i = 0; i < robot_names_.size(); ++i) {
      const auto & robot = robot_names_[i];
      const auto goal = plans_[robot].goal;
      if (!goal) {
        continue;
      }
      const Color color = colorForIndex(i);

      auto circle = baseMarker("goal_radius", id++, visualization_msgs::msg::Marker::LINE_STRIP);
      circle.scale.x = 0.035;
      circle.color.r = color.r;
      circle.color.g = color.g;
      circle.color.b = color.b;
      circle.color.a = 0.95f;
      for (int step = 0; step <= 72; ++step) {
        const double angle = 2.0 * 3.14159265358979323846 * static_cast<double>(step) / 72.0;
        geometry_msgs::msg::Point point;
        point.x = goal->point.x + planner_params_.goal_radius_m * std::cos(angle);
        point.y = goal->point.y + planner_params_.goal_radius_m * std::sin(angle);
        point.z = 0.03;
        circle.points.push_back(point);
      }
      markers.markers.push_back(circle);

      auto dot = baseMarker("goal_point", id++, visualization_msgs::msg::Marker::SPHERE);
      dot.pose.position.x = goal->point.x;
      dot.pose.position.y = goal->point.y;
      dot.pose.position.z = 0.05;
      dot.scale.x = 0.12;
      dot.scale.y = 0.12;
      dot.scale.z = 0.04;
      dot.color.r = color.r;
      dot.color.g = color.g;
      dot.color.b = color.b;
      dot.color.a = 0.8f;
      markers.markers.push_back(dot);

      auto text = baseMarker("goal_label", id++, visualization_msgs::msg::Marker::TEXT_VIEW_FACING);
      text.pose.position.x = goal->point.x;
      text.pose.position.y = goal->point.y;
      text.pose.position.z = 0.28;
      text.scale.z = 0.22;
      text.color.r = color.r;
      text.color.g = color.g;
      text.color.b = color.b;
      text.color.a = 1.0f;
      text.text = robot;
      markers.markers.push_back(text);
    }
    marker_pub_->publish(markers);
  }

  void publishPlanMarkersLocked()
  {
    visualization_msgs::msg::MarkerArray markers;
    addDeleteAll(markers);
    int id = 1;

    for (size_t i = 0; i < robot_names_.size(); ++i) {
      const auto & robot = robot_names_[i];
      const Color color = colorForIndex(i);

      if (plans_[robot].goal) {
        const auto & goal = *plans_[robot].goal;
        auto circle = baseMarker("goal_radius", id++, visualization_msgs::msg::Marker::LINE_STRIP);
        circle.scale.x = 0.035;
        circle.color.r = color.r;
        circle.color.g = color.g;
        circle.color.b = color.b;
        circle.color.a = 0.95f;
        for (int step = 0; step <= 72; ++step) {
          const double angle = 2.0 * 3.14159265358979323846 * static_cast<double>(step) / 72.0;
          geometry_msgs::msg::Point point;
          point.x = goal.point.x + planner_params_.goal_radius_m * std::cos(angle);
          point.y = goal.point.y + planner_params_.goal_radius_m * std::sin(angle);
          point.z = 0.03;
          circle.points.push_back(point);
        }
        markers.markers.push_back(circle);
      }

      if (!plans_[robot].path.poses.empty()) {
        auto path_marker = baseMarker("central_path", id++, visualization_msgs::msg::Marker::LINE_STRIP);
        path_marker.scale.x = 0.045;
        path_marker.color.r = color.r;
        path_marker.color.g = color.g;
        path_marker.color.b = color.b;
        path_marker.color.a = 0.95f;
        for (const auto & pose : plans_[robot].path.poses) {
          geometry_msgs::msg::Point point;
          point.x = pose.pose.position.x;
          point.y = pose.pose.position.y;
          point.z = 0.08;
          path_marker.points.push_back(point);
        }
        markers.markers.push_back(path_marker);
      }
    }

    marker_pub_->publish(markers);
  }

  visualization_msgs::msg::Marker baseMarker(
    const std::string & ns,
    int id,
    int32_t type)
  {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = map_frame_;
    marker.header.stamp = this->now();
    marker.ns = ns;
    marker.id = id;
    marker.type = type;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.05;
    marker.scale.y = 0.05;
    marker.scale.z = 0.05;
    marker.color.a = 1.0f;
    return marker;
  }

  void addDeleteAll(visualization_msgs::msg::MarkerArray & markers)
  {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = map_frame_;
    marker.header.stamp = this->now();
    marker.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(marker);
  }

  void publishDeleteMarkers()
  {
    visualization_msgs::msg::MarkerArray markers;
    addDeleteAll(markers);
    marker_pub_->publish(markers);
  }

  Color colorForIndex(size_t index) const
  {
    static const std::array<Color, 6> colors = {{
      {1.0f, 0.25f, 0.25f},
      {0.25f, 0.70f, 1.0f},
      {1.0f, 0.78f, 0.20f},
      {0.35f, 1.0f, 0.45f},
      {0.95f, 0.45f, 1.0f},
      {0.70f, 0.90f, 0.20f},
    }};
    return colors[index % colors.size()];
  }

  void publishStatus(const std::string & text)
  {
    std_msgs::msg::String msg;
    msg.data = text;
    status_pub_->publish(msg);
    RCLCPP_INFO(this->get_logger(), "%s", text.c_str());
  }

  void publishStatusLocked(const std::string & text)
  {
    std_msgs::msg::String msg;
    msg.data = text;
    status_pub_->publish(msg);
    RCLCPP_INFO(this->get_logger(), "%s", text.c_str());
  }

  std::vector<std::string> robot_names_;
  std::string map_topic_;
  std::string map_frame_;
  std::string base_frame_;
  std::string fallback_base_frame_;
  bool execute_after_all_goals_{true};
  double replan_deviation_m_{0.55};
  double path_min_spacing_m_{0.05};
  double follow_path_server_timeout_s_{2.0};
  PlannerParams planner_params_;

  tf2_ros::Buffer tf_buffer_;
  std::mutex mutex_;
  std::optional<nav_msgs::msg::OccupancyGrid> latest_map_;
  std::map<std::string, RobotPlan> plans_;
  std::map<std::string, nav_msgs::msg::Path::SharedPtr> last_paths_;
  std::map<std::string, rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr> path_pubs_;
  std::map<std::string, rclcpp_action::Client<FollowPath>::SharedPtr> follow_clients_;
  std::map<std::string, GoalHandleFollowPath::SharedPtr> active_handles_;
  std::map<std::string, rclcpp_action::ResultCode> completed_results_;

  bool plan_requested_{false};
  bool planning_{false};
  bool execution_active_{false};

  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  std::vector<rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr> goal_subs_;
  std::vector<rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr> tf_subs_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr clear_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr replan_srv_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace multi_robot_swarm_planner

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<multi_robot_swarm_planner::CentralFleetPlanner>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
