#include "multi_robot_swarm_planner/swarm_mppi.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

namespace multi_robot_swarm_planner
{
namespace
{

constexpr int kMaxRobots = 16;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kLambda = 6.0f;
constexpr float kNoiseStdV = 0.08f;
constexpr float kNoiseStdW = 0.60f;

struct DeviceControl
{
  float v;
  float w;
};

struct DeviceMap
{
  int width;
  int height;
  float resolution;
  float origin_x;
  float origin_y;
};

#define CUDA_CHECK_OR_RETURN(call, context_text)                                 \
  do {                                                                          \
    cudaError_t err = (call);                                                    \
    if (err != cudaSuccess) {                                                    \
      std::ostringstream oss;                                                    \
      oss << (context_text) << ": " << cudaGetErrorString(err);                 \
      result.success = false;                                                    \
      result.message = oss.str();                                                \
      return result;                                                             \
    }                                                                            \
  } while (0)

__host__ __device__ float clampf(float value, float low, float high)
{
  return fminf(fmaxf(value, low), high);
}

__host__ __device__ float normalizeAngle(float angle)
{
  while (angle > kPi) {
    angle -= 2.0f * kPi;
  }
  while (angle < -kPi) {
    angle += 2.0f * kPi;
  }
  return angle;
}

__host__ __device__ float distance2d(PlannerState a, PlannerState b)
{
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  return sqrtf(dx * dx + dy * dy);
}

__host__ __device__ PlannerState propagate(
  PlannerState state, DeviceControl control, float dt)
{
  state.x += control.v * cosf(state.theta) * dt;
  state.y += control.v * sinf(state.theta) * dt;
  state.theta = normalizeAngle(state.theta + control.w * dt);
  return state;
}

__host__ __device__ DeviceControl limitAccel(
  DeviceControl desired, DeviceControl previous, const PlannerParams params)
{
  DeviceControl out;
  out.v = previous.v + clampf(desired.v - previous.v, -params.max_dv_step, params.max_dv_step);
  out.w = previous.w + clampf(desired.w - previous.w, -params.max_dw_step, params.max_dw_step);
  out.v = clampf(out.v, 0.0f, params.max_v_mps);
  out.w = clampf(out.w, -params.max_w_radps, params.max_w_radps);
  return out;
}

__host__ __device__ DeviceControl goalController(
  PlannerState state, PlannerState goal, const PlannerParams params)
{
  const float dx = goal.x - state.x;
  const float dy = goal.y - state.y;
  const float dist = sqrtf(dx * dx + dy * dy);

  DeviceControl control{0.0f, 0.0f};
  if (dist <= params.goal_radius_m) {
    return control;
  }

  const float desired_heading = atan2f(dy, dx);
  const float heading_error = normalizeAngle(desired_heading - state.theta);
  float speed = clampf(0.55f * dist, 0.0f, params.max_v_mps);
  if (fabsf(heading_error) > 0.75f) {
    speed *= 0.35f;
  }
  if (fabsf(heading_error) > 1.45f) {
    speed = 0.0f;
  }

  control.v = speed;
  control.w = clampf(2.0f * heading_error, -params.max_w_radps, params.max_w_radps);
  return control;
}

__device__ bool worldToMap(
  const DeviceMap map, float x, float y, int * mx, int * my)
{
  *mx = static_cast<int>(floorf((x - map.origin_x) / map.resolution));
  *my = static_cast<int>(floorf((y - map.origin_y) / map.resolution));
  return *mx >= 0 && *mx < map.width && *my >= 0 && *my < map.height;
}

__device__ float mapCostAt(
  const DeviceMap map,
  const int8_t * data,
  PlannerState state,
  const PlannerParams params)
{
  const float offsets[9][2] = {
    {0.0f, 0.0f},
    {params.robot_radius_m, 0.0f},
    {-params.robot_radius_m, 0.0f},
    {0.0f, params.robot_radius_m},
    {0.0f, -params.robot_radius_m},
    {0.7071f * params.robot_radius_m, 0.7071f * params.robot_radius_m},
    {0.7071f * params.robot_radius_m, -0.7071f * params.robot_radius_m},
    {-0.7071f * params.robot_radius_m, 0.7071f * params.robot_radius_m},
    {-0.7071f * params.robot_radius_m, -0.7071f * params.robot_radius_m},
  };

  float total = 0.0f;
  for (int i = 0; i < 9; ++i) {
    int mx = 0;
    int my = 0;
    if (!worldToMap(map, state.x + offsets[i][0], state.y + offsets[i][1], &mx, &my)) {
      total += 100000.0f;
      continue;
    }
    const int8_t value = data[my * map.width + mx];
    if (value < 0) {
      total += params.treat_unknown_as_obstacle ? 50000.0f : 5.0f;
    } else if (value >= params.occupancy_threshold) {
      total += 100000.0f + 500.0f * static_cast<float>(value);
    } else {
      total += 0.02f * static_cast<float>(value);
    }
  }
  return total;
}

__global__ void initRngKernel(curandState * states, unsigned long long seed, int samples)
{
  const int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample_id >= samples) {
    return;
  }
  curand_init(seed, sample_id, 0, &states[sample_id]);
}

__global__ void rolloutKernel(
  const PlannerState * current_states,
  const PlannerState * goals,
  const DeviceControl * last_controls,
  const DeviceControl * nominal_controls,
  const int8_t * map_data,
  DeviceMap map,
  PlannerParams params,
  curandState * rng_states,
  float * costs,
  DeviceControl * sampled_controls,
  int num_robots,
  int samples)
{
  const int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample_id >= samples) {
    return;
  }

  curandState rng = rng_states[sample_id];
  PlannerState states[kMaxRobots];
  DeviceControl previous[kMaxRobots];

  for (int robot = 0; robot < num_robots; ++robot) {
    states[robot] = current_states[robot];
    previous[robot] = last_controls[robot];
  }

  float cost = 0.0f;
  for (int t = 0; t < params.horizon; ++t) {
    DeviceControl controls[kMaxRobots];
    for (int robot = 0; robot < num_robots; ++robot) {
      DeviceControl nominal = nominal_controls[robot * params.horizon + t];
      DeviceControl noisy;
      noisy.v = nominal.v + curand_normal(&rng) * kNoiseStdV;
      noisy.w = nominal.w + curand_normal(&rng) * kNoiseStdW;
      noisy.v = clampf(noisy.v, 0.0f, params.max_v_mps);
      noisy.w = clampf(noisy.w, -params.max_w_radps, params.max_w_radps);
      if (distance2d(states[robot], goals[robot]) <= params.goal_radius_m) {
        noisy.v = 0.0f;
        noisy.w = 0.0f;
      }

      controls[robot] = limitAccel(noisy, previous[robot], params);
      previous[robot] = controls[robot];
      sampled_controls[(sample_id * num_robots + robot) * params.horizon + t] = controls[robot];
    }

    for (int robot = 0; robot < num_robots; ++robot) {
      states[robot] = propagate(states[robot], controls[robot], params.dt);

      const float dist = distance2d(states[robot], goals[robot]);
      const float remaining = fmaxf(0.0f, dist - params.goal_radius_m);
      cost += 4.0f * remaining;
      cost += 0.06f * controls[robot].v * controls[robot].v;
      cost += 0.02f * controls[robot].w * controls[robot].w;
      cost += mapCostAt(map, map_data, states[robot], params);
    }

    for (int a = 0; a < num_robots; ++a) {
      for (int b = a + 1; b < num_robots; ++b) {
        const float d = distance2d(states[a], states[b]);
        if (d < params.safety_distance_m) {
          const float lack = params.safety_distance_m - d;
          cost += 200000.0f * lack * lack + 1000.0f / (d + 0.02f);
        } else if (d < params.planning_clearance_m) {
          const float lack = params.planning_clearance_m - d;
          cost += 1500.0f * lack * lack;
        }
      }
    }
  }

  for (int robot = 0; robot < num_robots; ++robot) {
    const float terminal = fmaxf(0.0f, distance2d(states[robot], goals[robot]) - params.goal_radius_m);
    cost += 80.0f * terminal;
  }

  rng_states[sample_id] = rng;
  costs[sample_id] = cost;
}

__global__ void updateKernel(
  const float * costs,
  const DeviceControl * sampled_controls,
  DeviceControl * nominal_controls,
  PlannerParams params,
  int num_robots,
  int samples)
{
  const int robot = blockIdx.x;
  const int t = threadIdx.x;

  __shared__ float min_cost;
  __shared__ float sum_weights;

  if (t == 0) {
    float best = 1e30f;
    for (int k = 0; k < samples; ++k) {
      best = fminf(best, costs[k]);
    }
    min_cost = best;

    float total = 0.0f;
    for (int k = 0; k < samples; ++k) {
      total += expf(-(costs[k] - min_cost) / kLambda);
    }
    sum_weights = total;
  }

  __syncthreads();

  if (robot >= num_robots || t >= params.horizon) {
    return;
  }

  float weighted_v = 0.0f;
  float weighted_w = 0.0f;
  for (int k = 0; k < samples; ++k) {
    const float weight = expf(-(costs[k] - min_cost) / kLambda) / (sum_weights + 1e-8f);
    const DeviceControl control = sampled_controls[(k * num_robots + robot) * params.horizon + t];
    weighted_v += weight * control.v;
    weighted_w += weight * control.w;
  }

  nominal_controls[robot * params.horizon + t].v = clampf(weighted_v, 0.0f, params.max_v_mps);
  nominal_controls[robot * params.horizon + t].w = clampf(weighted_w, -params.max_w_radps, params.max_w_radps);
}

bool reachedAll(
  const std::vector<PlannerState> & states,
  const std::vector<PlannerState> & goals,
  const PlannerParams & params)
{
  for (size_t i = 0; i < states.size(); ++i) {
    if (distance2d(states[i], goals[i]) > params.goal_radius_m) {
      return false;
    }
  }
  return true;
}

bool inMapAndFree(const PlannerMap & map, PlannerState state, const PlannerParams & params)
{
  const float offsets[9][2] = {
    {0.0f, 0.0f},
    {params.robot_radius_m, 0.0f},
    {-params.robot_radius_m, 0.0f},
    {0.0f, params.robot_radius_m},
    {0.0f, -params.robot_radius_m},
    {0.7071f * params.robot_radius_m, 0.7071f * params.robot_radius_m},
    {0.7071f * params.robot_radius_m, -0.7071f * params.robot_radius_m},
    {-0.7071f * params.robot_radius_m, 0.7071f * params.robot_radius_m},
    {-0.7071f * params.robot_radius_m, -0.7071f * params.robot_radius_m},
  };

  for (auto & offset : offsets) {
    const int mx = static_cast<int>(std::floor((state.x + offset[0] - map.origin_x) / map.resolution));
    const int my = static_cast<int>(std::floor((state.y + offset[1] - map.origin_y) / map.resolution));
    if (mx < 0 || mx >= map.width || my < 0 || my >= map.height) {
      return false;
    }
    const int8_t value = map.data[my * map.width + mx];
    if (value < 0 && params.treat_unknown_as_obstacle) {
      return false;
    }
    if (value >= params.occupancy_threshold) {
      return false;
    }
  }
  return true;
}

void seedNominalControls(
  const std::vector<PlannerState> & states,
  const std::vector<PlannerState> & goals,
  std::vector<DeviceControl> & nominal,
  const PlannerParams & params)
{
  for (size_t robot = 0; robot < states.size(); ++robot) {
    PlannerState predicted = states[robot];
    DeviceControl previous{0.0f, 0.0f};
    for (int t = 0; t < params.horizon; ++t) {
      DeviceControl control = goalController(predicted, goals[robot], params);
      control = limitAccel(control, previous, params);
      nominal[robot * params.horizon + t] = control;
      predicted = propagate(predicted, control, params.dt);
      previous = control;
    }
  }
}

void shiftNominalControls(
  const std::vector<PlannerState> & states,
  const std::vector<PlannerState> & goals,
  std::vector<DeviceControl> & nominal,
  const PlannerParams & params)
{
  const int num_robots = static_cast<int>(states.size());
  for (int robot = 0; robot < num_robots; ++robot) {
    for (int t = 0; t < params.horizon - 1; ++t) {
      nominal[robot * params.horizon + t] = nominal[robot * params.horizon + t + 1];
    }
    nominal[robot * params.horizon + params.horizon - 1] =
      goalController(states[robot], goals[robot], params);
  }
}

}  // namespace

PlanResult planSwarmMppi(
  const PlannerMap & map,
  const std::vector<PlannerState> & starts,
  const std::vector<PlannerState> & goals,
  const PlannerParams & raw_params)
{
  PlanResult result;
  PlannerParams params = raw_params;
  params.horizon = std::max(2, params.horizon);
  params.samples = std::max(16, params.samples);
  params.mppi_iterations = std::max(1, params.mppi_iterations);
  params.max_plan_steps = std::max(1, params.max_plan_steps);

  const int num_robots = static_cast<int>(starts.size());
  if (num_robots <= 0 || num_robots > kMaxRobots || starts.size() != goals.size()) {
    result.message = "invalid robot count for MPPI planner";
    return result;
  }
  if (map.width <= 0 || map.height <= 0 || map.data.size() != static_cast<size_t>(map.width * map.height)) {
    result.message = "invalid occupancy grid for MPPI planner";
    return result;
  }

  for (int robot = 0; robot < num_robots; ++robot) {
    if (!inMapAndFree(map, starts[robot], params)) {
      result.message = "robot start is outside map or occupied";
      return result;
    }
    if (!inMapAndFree(map, goals[robot], params)) {
      result.message = "robot goal is outside map or occupied";
      return result;
    }
  }

  result.paths.resize(num_robots);
  std::vector<PlannerState> states = starts;
  std::vector<DeviceControl> last_controls(num_robots, {0.0f, 0.0f});
  std::vector<DeviceControl> controls(num_robots, {0.0f, 0.0f});
  std::vector<DeviceControl> nominal(num_robots * params.horizon);
  seedNominalControls(states, goals, nominal, params);

  for (int robot = 0; robot < num_robots; ++robot) {
    result.paths[robot].push_back(states[robot]);
  }

  PlannerState * d_states = nullptr;
  PlannerState * d_goals = nullptr;
  DeviceControl * d_last_controls = nullptr;
  DeviceControl * d_nominal = nullptr;
  DeviceControl * d_sampled = nullptr;
  int8_t * d_map = nullptr;
  curandState * d_rng = nullptr;
  float * d_costs = nullptr;

  CUDA_CHECK_OR_RETURN(cudaMalloc(&d_states, num_robots * sizeof(PlannerState)), "cudaMalloc states");
  CUDA_CHECK_OR_RETURN(cudaMalloc(&d_goals, num_robots * sizeof(PlannerState)), "cudaMalloc goals");
  CUDA_CHECK_OR_RETURN(cudaMalloc(&d_last_controls, num_robots * sizeof(DeviceControl)), "cudaMalloc controls");
  CUDA_CHECK_OR_RETURN(cudaMalloc(&d_nominal, nominal.size() * sizeof(DeviceControl)), "cudaMalloc nominal");
  CUDA_CHECK_OR_RETURN(
    cudaMalloc(&d_sampled, static_cast<size_t>(params.samples) * num_robots * params.horizon * sizeof(DeviceControl)),
    "cudaMalloc sampled controls");
  CUDA_CHECK_OR_RETURN(cudaMalloc(&d_map, map.data.size() * sizeof(int8_t)), "cudaMalloc map");
  CUDA_CHECK_OR_RETURN(cudaMalloc(&d_rng, params.samples * sizeof(curandState)), "cudaMalloc rng");
  CUDA_CHECK_OR_RETURN(cudaMalloc(&d_costs, params.samples * sizeof(float)), "cudaMalloc costs");

  CUDA_CHECK_OR_RETURN(cudaMemcpy(d_goals, goals.data(), num_robots * sizeof(PlannerState), cudaMemcpyHostToDevice), "copy goals");
  CUDA_CHECK_OR_RETURN(cudaMemcpy(d_map, map.data.data(), map.data.size() * sizeof(int8_t), cudaMemcpyHostToDevice), "copy map");
  DeviceMap device_map{map.width, map.height, map.resolution, map.origin_x, map.origin_y};

  const int threads = 256;
  const int blocks = (params.samples + threads - 1) / threads;
  initRngKernel<<<blocks, threads>>>(d_rng, 1234ULL, params.samples);
  CUDA_CHECK_OR_RETURN(cudaGetLastError(), "launch init rng");
  CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize(), "sync init rng");

  int completed_step = 0;
  for (int step = 1; step <= params.max_plan_steps; ++step) {
    CUDA_CHECK_OR_RETURN(cudaMemcpy(d_states, states.data(), num_robots * sizeof(PlannerState), cudaMemcpyHostToDevice), "copy states");
    CUDA_CHECK_OR_RETURN(cudaMemcpy(d_last_controls, last_controls.data(), num_robots * sizeof(DeviceControl), cudaMemcpyHostToDevice), "copy last controls");
    CUDA_CHECK_OR_RETURN(cudaMemcpy(d_nominal, nominal.data(), nominal.size() * sizeof(DeviceControl), cudaMemcpyHostToDevice), "copy nominal");

    for (int iter = 0; iter < params.mppi_iterations; ++iter) {
      rolloutKernel<<<blocks, threads>>>(
        d_states,
        d_goals,
        d_last_controls,
        d_nominal,
        d_map,
        device_map,
        params,
        d_rng,
        d_costs,
        d_sampled,
        num_robots,
        params.samples);
      CUDA_CHECK_OR_RETURN(cudaGetLastError(), "launch rollout");
      CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize(), "sync rollout");

      updateKernel<<<num_robots, params.horizon>>>(
        d_costs,
        d_sampled,
        d_nominal,
        params,
        num_robots,
        params.samples);
      CUDA_CHECK_OR_RETURN(cudaGetLastError(), "launch update");
      CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize(), "sync update");
    }

    CUDA_CHECK_OR_RETURN(cudaMemcpy(nominal.data(), d_nominal, nominal.size() * sizeof(DeviceControl), cudaMemcpyDeviceToHost), "copy updated nominal");
    for (int robot = 0; robot < num_robots; ++robot) {
      controls[robot] = nominal[robot * params.horizon];
      if (distance2d(states[robot], goals[robot]) <= params.goal_radius_m) {
        controls[robot] = {0.0f, 0.0f};
      }
    }

    bool unsafe = false;
    std::vector<PlannerState> proposed = states;
    for (int robot = 0; robot < num_robots; ++robot) {
      proposed[robot] = propagate(states[robot], controls[robot], params.dt);
      if (!inMapAndFree(map, proposed[robot], params)) {
        controls[robot].v = 0.0f;
        proposed[robot] = propagate(states[robot], controls[robot], params.dt);
      }
      if (!inMapAndFree(map, proposed[robot], params)) {
        unsafe = true;
      }
    }

    for (int a = 0; a < num_robots; ++a) {
      for (int b = a + 1; b < num_robots; ++b) {
        if (distance2d(proposed[a], proposed[b]) < params.safety_distance_m) {
          const float da = distance2d(states[a], goals[a]);
          const float db = distance2d(states[b], goals[b]);
          const int stop_robot = da > db ? a : b;
          controls[stop_robot].v = 0.0f;
          proposed[stop_robot] = propagate(states[stop_robot], controls[stop_robot], params.dt);
        }
      }
    }

    if (unsafe) {
      result.message = "planner generated an unsafe map collision";
      break;
    }

    for (int robot = 0; robot < num_robots; ++robot) {
      states[robot] = proposed[robot];
      last_controls[robot] = controls[robot];
      result.paths[robot].push_back(states[robot]);
    }
    completed_step = step;

    if (reachedAll(states, goals, params)) {
      result.success = true;
      result.message = "planned coordinated paths";
      break;
    }

    shiftNominalControls(states, goals, nominal, params);
  }

  if (!result.success) {
    const bool near_enough = reachedAll(states, goals, params);
    result.success = near_enough;
    if (near_enough) {
      result.message = "planned coordinated paths";
    } else if (result.message.empty()) {
      std::ostringstream oss;
      oss << "planner hit max_plan_steps=" << params.max_plan_steps
          << " after " << completed_step << " steps";
      result.message = oss.str();
    }
  }

  cudaFree(d_states);
  cudaFree(d_goals);
  cudaFree(d_last_controls);
  cudaFree(d_nominal);
  cudaFree(d_sampled);
  cudaFree(d_map);
  cudaFree(d_rng);
  cudaFree(d_costs);

  return result;
}

}  // namespace multi_robot_swarm_planner
