#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace multi_robot_swarm_planner
{

struct PlannerState
{
  float x{0.0f};
  float y{0.0f};
  float theta{0.0f};
};

struct PlannerMap
{
  int width{0};
  int height{0};
  float resolution{0.05f};
  float origin_x{0.0f};
  float origin_y{0.0f};
  std::vector<int8_t> data;
};

struct PlannerParams
{
  float dt{0.1f};
  float goal_radius_m{0.25f};
  float robot_radius_m{0.22f};
  float safety_distance_m{0.45f};
  float planning_clearance_m{0.55f};
  float max_v_mps{0.20f};
  float max_w_radps{1.20f};
  float max_dv_step{0.03f};
  float max_dw_step{0.18f};
  int horizon{80};
  int samples{512};
  int mppi_iterations{6};
  int max_plan_steps{900};
  int occupancy_threshold{50};
  bool treat_unknown_as_obstacle{true};
};

struct PlanResult
{
  bool success{false};
  std::string message;
  std::vector<std::vector<PlannerState>> paths;
};

PlanResult planSwarmMppi(
  const PlannerMap & map,
  const std::vector<PlannerState> & starts,
  const std::vector<PlannerState> & goals,
  const PlannerParams & params);

}  // namespace multi_robot_swarm_planner
