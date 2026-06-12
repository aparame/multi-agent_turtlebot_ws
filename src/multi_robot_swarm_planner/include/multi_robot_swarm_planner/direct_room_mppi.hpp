#pragma once

#include <string>
#include <vector>

namespace multi_robot_swarm_planner
{

struct DirectMppiState
{
  float x{0.0f};
  float y{0.0f};
  float theta{0.0f};
};

struct DirectMppiControl
{
  float v{0.0f};
  float omega{0.0f};
};

struct DirectMppiParams
{
  float dt{0.1f};
  float room_size_m{3.048f};
  float goal_radius_m{0.12f};
  float max_v_mps{0.50f};
  float max_w_radps{0.20f};
  float max_dv_step{0.03f};
  float max_dw_step{0.18f};
  float safety_distance_m{0.25f};
  float planning_clearance_m{0.45f};
  float wall_margin_m{0.20f};
  int horizon{80};
  int samples{512};
  int mppi_iterations{6};
};

struct DirectMppiResult
{
  bool success{false};
  bool all_reached{false};
  std::string message;
  std::vector<DirectMppiControl> controls;
  std::vector<std::vector<DirectMppiState>> preview_paths;
};

DirectMppiResult computeDirectRoomMppi(
  const std::vector<DirectMppiState> & states,
  const std::vector<DirectMppiState> & goals,
  const std::vector<DirectMppiControl> & last_controls,
  const DirectMppiParams & raw_params);

}  // namespace multi_robot_swarm_planner
