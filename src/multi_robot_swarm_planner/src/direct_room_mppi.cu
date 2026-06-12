#include "multi_robot_swarm_planner/direct_room_mppi.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cmath>
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

__host__ __device__ float distance2d(DirectMppiState a, DirectMppiState b)
{
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  return sqrtf(dx * dx + dy * dy);
}

__host__ __device__ bool insideRoom(DirectMppiState state, DirectMppiParams params)
{
  const float half = fmaxf(0.0f, 0.5f * params.room_size_m - params.wall_margin_m);
  return state.x >= -half && state.x <= half && state.y >= -half && state.y <= half;
}

__host__ __device__ DirectMppiState propagate(
  DirectMppiState state, DirectMppiControl control, DirectMppiParams params)
{
  state.x += control.v * cosf(state.theta) * params.dt;
  state.y += control.v * sinf(state.theta) * params.dt;
  state.theta = normalizeAngle(state.theta + control.omega * params.dt);
  return state;
}

__host__ __device__ DirectMppiControl goalController(
  DirectMppiState state, DirectMppiState goal, DirectMppiParams params)
{
  const float dx = goal.x - state.x;
  const float dy = goal.y - state.y;
  const float dist = sqrtf(dx * dx + dy * dy);

  DirectMppiControl control{0.0f, 0.0f};
  if (dist < params.goal_radius_m) {
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
  control.omega = clampf(2.0f * heading_error, -params.max_w_radps, params.max_w_radps);
  return control;
}

__host__ __device__ DirectMppiControl limitAccel(
  DirectMppiControl desired, DirectMppiControl previous, DirectMppiParams params)
{
  DirectMppiControl out;
  out.v = previous.v + clampf(desired.v - previous.v, -params.max_dv_step, params.max_dv_step);
  out.omega = previous.omega +
    clampf(desired.omega - previous.omega, -params.max_dw_step, params.max_dw_step);
  out.v = clampf(out.v, 0.0f, params.max_v_mps);
  out.omega = clampf(out.omega, -params.max_w_radps, params.max_w_radps);
  return out;
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
  const DirectMppiState * current_states,
  const DirectMppiState * goals,
  const DirectMppiControl * last_controls,
  const DirectMppiControl * nominal_controls,
  curandState * rng_states,
  float * costs,
  DirectMppiControl * sampled_controls,
  DirectMppiParams params,
  int num_robots,
  int samples)
{
  const int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample_id >= samples) {
    return;
  }

  curandState rng = rng_states[sample_id];
  DirectMppiState states[kMaxRobots];
  DirectMppiControl previous[kMaxRobots];

  for (int robot = 0; robot < num_robots; ++robot) {
    states[robot] = current_states[robot];
    previous[robot] = last_controls[robot];
  }

  float cost = 0.0f;
  const float half_room = 0.5f * params.room_size_m;
  const float safe_half_room = fmaxf(0.0f, 0.5f * params.room_size_m - params.wall_margin_m);

  for (int t = 0; t < params.horizon; ++t) {
    DirectMppiControl controls[kMaxRobots];

    for (int robot = 0; robot < num_robots; ++robot) {
      DirectMppiControl nominal = nominal_controls[robot * params.horizon + t];
      DirectMppiControl noisy;
      noisy.v = nominal.v + curand_normal(&rng) * kNoiseStdV;
      noisy.omega = nominal.omega + curand_normal(&rng) * kNoiseStdW;
      noisy.v = clampf(noisy.v, 0.0f, params.max_v_mps);
      noisy.omega = clampf(noisy.omega, -params.max_w_radps, params.max_w_radps);
      if (distance2d(states[robot], goals[robot]) < params.goal_radius_m) {
        noisy.v = 0.0f;
        noisy.omega = 0.0f;
      }
      controls[robot] = limitAccel(noisy, previous[robot], params);
      previous[robot] = controls[robot];
      sampled_controls[(sample_id * num_robots + robot) * params.horizon + t] = controls[robot];
    }

    for (int robot = 0; robot < num_robots; ++robot) {
      states[robot] = propagate(states[robot], controls[robot], params);
      const float dist = distance2d(states[robot], goals[robot]);
      cost += 3.0f * dist;
      cost += 0.08f * controls[robot].v * controls[robot].v;
      cost += 0.03f * controls[robot].omega * controls[robot].omega;

      if (!insideRoom(states[robot], params)) {
        const float over_x = fmaxf(0.0f, -safe_half_room - states[robot].x) +
          fmaxf(0.0f, states[robot].x - safe_half_room);
        const float over_y = fmaxf(0.0f, -safe_half_room - states[robot].y) +
          fmaxf(0.0f, states[robot].y - safe_half_room);
        cost += 100000.0f + 50000.0f * (over_x + over_y);
      } else {
        const float margin = fminf(
          fminf(states[robot].x + half_room, half_room - states[robot].x),
          fminf(states[robot].y + half_room, half_room - states[robot].y));
        if (margin < params.wall_margin_m) {
          const float lack = params.wall_margin_m - margin;
          cost += 80.0f * lack * lack;
        }
      }
    }

    for (int a = 0; a < num_robots; ++a) {
      for (int b = a + 1; b < num_robots; ++b) {
        const float d = distance2d(states[a], states[b]);
        if (d < params.planning_clearance_m) {
          const float lack = params.planning_clearance_m - d;
          cost += 150000.0f * lack * lack + 500.0f / (d + 0.02f);
        } else if (d < params.planning_clearance_m * 1.7f) {
          const float lack = params.planning_clearance_m * 1.7f - d;
          cost += 20.0f * lack * lack;
        }
      }
    }
  }

  for (int robot = 0; robot < num_robots; ++robot) {
    const float terminal_dist = distance2d(states[robot], goals[robot]);
    cost += 70.0f * terminal_dist;
  }

  rng_states[sample_id] = rng;
  costs[sample_id] = cost;
}

__global__ void updateKernel(
  const float * costs,
  const DirectMppiControl * sampled_controls,
  DirectMppiControl * nominal_controls,
  DirectMppiParams params,
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
  float weighted_omega = 0.0f;
  for (int k = 0; k < samples; ++k) {
    const float weight = expf(-(costs[k] - min_cost) / kLambda) / (sum_weights + 1e-8f);
    const DirectMppiControl control = sampled_controls[(k * num_robots + robot) *
      params.horizon + t];
    weighted_v += weight * control.v;
    weighted_omega += weight * control.omega;
  }

  nominal_controls[robot * params.horizon + t].v = clampf(
    weighted_v, 0.0f, params.max_v_mps);
  nominal_controls[robot * params.horizon + t].omega = clampf(
    weighted_omega, -params.max_w_radps, params.max_w_radps);
}

bool allReached(
  const std::vector<DirectMppiState> & states,
  const std::vector<DirectMppiState> & goals,
  const DirectMppiParams & params)
{
  for (size_t i = 0; i < states.size(); ++i) {
    if (distance2d(states[i], goals[i]) > params.goal_radius_m) {
      return false;
    }
  }
  return true;
}

void seedNominalControls(
  const std::vector<DirectMppiState> & states,
  const std::vector<DirectMppiState> & goals,
  std::vector<DirectMppiControl> & nominal,
  const DirectMppiParams & params)
{
  for (size_t robot = 0; robot < states.size(); ++robot) {
    DirectMppiState predicted = states[robot];
    DirectMppiControl previous{0.0f, 0.0f};
    for (int t = 0; t < params.horizon; ++t) {
      DirectMppiControl control = goalController(predicted, goals[robot], params);
      control = limitAccel(control, previous, params);
      nominal[robot * params.horizon + t] = control;
      predicted = propagate(predicted, control, params);
      previous = control;
    }
  }
}

}  // namespace

DirectMppiResult computeDirectRoomMppi(
  const std::vector<DirectMppiState> & states,
  const std::vector<DirectMppiState> & goals,
  const std::vector<DirectMppiControl> & last_controls,
  const DirectMppiParams & raw_params)
{
  DirectMppiResult result;
  DirectMppiParams params = raw_params;
  params.horizon = std::max(2, std::min(params.horizon, 1024));
  params.samples = std::max(16, params.samples);
  params.mppi_iterations = std::max(1, params.mppi_iterations);

  const int num_robots = static_cast<int>(states.size());
  if (num_robots <= 0 || num_robots > kMaxRobots ||
    states.size() != goals.size() || states.size() != last_controls.size())
  {
    result.message = "invalid robot count for direct MPPI";
    return result;
  }

  for (int robot = 0; robot < num_robots; ++robot) {
    if (!insideRoom(states[robot], params)) {
      result.message = "robot state is outside the calibrated room";
      return result;
    }
    if (!insideRoom(goals[robot], params)) {
      result.message = "robot goal is outside the calibrated room";
      return result;
    }
  }

  result.controls.assign(num_robots, DirectMppiControl{0.0f, 0.0f});
  result.preview_paths.assign(num_robots, {});
  result.all_reached = allReached(states, goals, params);
  if (result.all_reached) {
    result.success = true;
    result.message = "all goals reached";
    for (int robot = 0; robot < num_robots; ++robot) {
      result.preview_paths[robot].push_back(states[robot]);
    }
    return result;
  }

  std::vector<DirectMppiControl> nominal(num_robots * params.horizon);
  seedNominalControls(states, goals, nominal, params);

  DirectMppiState * d_states = nullptr;
  DirectMppiState * d_goals = nullptr;
  DirectMppiControl * d_last_controls = nullptr;
  DirectMppiControl * d_nominal = nullptr;
  DirectMppiControl * d_sampled = nullptr;
  curandState * d_rng = nullptr;
  float * d_costs = nullptr;

  auto freeCuda = [&]() {
      cudaFree(d_states);
      cudaFree(d_goals);
      cudaFree(d_last_controls);
      cudaFree(d_nominal);
      cudaFree(d_sampled);
      cudaFree(d_rng);
      cudaFree(d_costs);
    };

  auto failCuda = [&](cudaError_t err, const char * context) {
      std::ostringstream oss;
      oss << context << ": " << cudaGetErrorString(err);
      result.success = false;
      result.message = oss.str();
      freeCuda();
      return result;
    };

  cudaError_t err = cudaSuccess;
  err = cudaMallocManaged(&d_states, num_robots * sizeof(DirectMppiState));
  if (err != cudaSuccess) {return failCuda(err, "cudaMalloc states");}
  err = cudaMallocManaged(&d_goals, num_robots * sizeof(DirectMppiState));
  if (err != cudaSuccess) {return failCuda(err, "cudaMalloc goals");}
  err = cudaMallocManaged(&d_last_controls, num_robots * sizeof(DirectMppiControl));
  if (err != cudaSuccess) {return failCuda(err, "cudaMalloc last controls");}
  err = cudaMallocManaged(&d_nominal, nominal.size() * sizeof(DirectMppiControl));
  if (err != cudaSuccess) {return failCuda(err, "cudaMalloc nominal");}
  err = cudaMallocManaged(
    &d_sampled,
    static_cast<size_t>(params.samples) * num_robots * params.horizon *
    sizeof(DirectMppiControl));
  if (err != cudaSuccess) {return failCuda(err, "cudaMalloc sampled controls");}
  err = cudaMallocManaged(&d_rng, params.samples * sizeof(curandState));
  if (err != cudaSuccess) {return failCuda(err, "cudaMalloc rng");}
  err = cudaMallocManaged(&d_costs, params.samples * sizeof(float));
  if (err != cudaSuccess) {return failCuda(err, "cudaMalloc costs");}

  for (int robot = 0; robot < num_robots; ++robot) {
    d_states[robot] = states[robot];
    d_goals[robot] = goals[robot];
    d_last_controls[robot] = last_controls[robot];
  }
  for (size_t i = 0; i < nominal.size(); ++i) {
    d_nominal[i] = nominal[i];
  }

  const int threads = 256;
  const int blocks = (params.samples + threads - 1) / threads;
  initRngKernel<<<blocks, threads>>>(d_rng, 1234ULL, params.samples);
  err = cudaGetLastError();
  if (err != cudaSuccess) {return failCuda(err, "launch init rng");}
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {return failCuda(err, "sync init rng");}

  for (int iter = 0; iter < params.mppi_iterations; ++iter) {
    rolloutKernel<<<blocks, threads>>>(
      d_states,
      d_goals,
      d_last_controls,
      d_nominal,
      d_rng,
      d_costs,
      d_sampled,
      params,
      num_robots,
      params.samples);
    err = cudaGetLastError();
    if (err != cudaSuccess) {return failCuda(err, "launch rollout");}
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {return failCuda(err, "sync rollout");}

    updateKernel<<<num_robots, params.horizon>>>(
      d_costs,
      d_sampled,
      d_nominal,
      params,
      num_robots,
      params.samples);
    err = cudaGetLastError();
    if (err != cudaSuccess) {return failCuda(err, "launch update");}
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {return failCuda(err, "sync update");}
  }

  for (int robot = 0; robot < num_robots; ++robot) {
    result.controls[robot] = d_nominal[robot * params.horizon];
    if (distance2d(states[robot], goals[robot]) <= params.goal_radius_m) {
      result.controls[robot] = DirectMppiControl{0.0f, 0.0f};
    }
    result.controls[robot] = limitAccel(result.controls[robot], last_controls[robot], params);
  }

  std::vector<DirectMppiState> proposed = states;
  for (int robot = 0; robot < num_robots; ++robot) {
    proposed[robot] = propagate(states[robot], result.controls[robot], params);
    if (!insideRoom(proposed[robot], params)) {
      result.controls[robot].v = 0.0f;
      proposed[robot] = propagate(states[robot], result.controls[robot], params);
    }
  }

  for (int a = 0; a < num_robots; ++a) {
    for (int b = a + 1; b < num_robots; ++b) {
      if (distance2d(proposed[a], proposed[b]) < params.safety_distance_m) {
        const float da = distance2d(states[a], goals[a]);
        const float db = distance2d(states[b], goals[b]);
        const int stop_robot = da > db ? a : b;
        result.controls[stop_robot].v = 0.0f;
      }
    }
  }

  for (int robot = 0; robot < num_robots; ++robot) {
    DirectMppiState preview = states[robot];
    result.preview_paths[robot].push_back(preview);
    for (int t = 0; t < params.horizon; ++t) {
      DirectMppiControl control = d_nominal[robot * params.horizon + t];
      if (distance2d(preview, goals[robot]) <= params.goal_radius_m) {
        control = DirectMppiControl{0.0f, 0.0f};
      }
      preview = propagate(preview, control, params);
      if (!insideRoom(preview, params)) {
        break;
      }
      result.preview_paths[robot].push_back(preview);
    }
  }

  result.success = true;
  result.message = "computed direct MPPI controls";
  freeCuda();
  return result;
}

}  // namespace multi_robot_swarm_planner
