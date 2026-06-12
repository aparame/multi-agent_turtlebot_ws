cmake_minimum_required(VERSION 3.18)
project(multi_robot_swarm_planner LANGUAGES CXX CUDA)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 86)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(action_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)

add_library(swarm_mppi
  src/swarm_mppi.cu
  src/direct_room_mppi.cu)
target_include_directories(swarm_mppi
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
set_target_properties(swarm_mppi PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON)

add_executable(central_fleet_planner src/central_fleet_planner.cpp)
target_include_directories(central_fleet_planner
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
ament_target_dependencies(central_fleet_planner
  action_msgs
  geometry_msgs
  nav2_msgs
  nav_msgs
  rclcpp
  rclcpp_action
  std_msgs
  std_srvs
  tf2
  tf2_geometry_msgs
  tf2_msgs
  tf2_ros
  visualization_msgs)
target_link_libraries(central_fleet_planner swarm_mppi)

add_executable(mppi_direct_controller src/mppi_direct_controller.cpp)
target_include_directories(mppi_direct_controller
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
ament_target_dependencies(mppi_direct_controller
  geometry_msgs
  nav_msgs
  rclcpp
  std_msgs
  std_srvs
  tf2
  tf2_geometry_msgs)
target_link_libraries(mppi_direct_controller swarm_mppi)

install(DIRECTORY include/
  DESTINATION include)

install(TARGETS
  swarm_mppi
  central_fleet_planner
  mppi_direct_controller
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})

ament_package()
