cmake_minimum_required(VERSION 3.8)
project(saturated_gyro_speed_estimation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -march=native)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(steam REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Boost COMPONENTS system REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(imu_publisher_node src/imu_publisher.cpp)
ament_target_dependencies(imu_publisher_node
  rclcpp
  std_msgs
  sensor_msgs
  steam
)
target_link_libraries(imu_publisher_node
  Eigen3::Eigen
  OpenMP::OpenMP_CXX
)

install(TARGETS imu_publisher_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
