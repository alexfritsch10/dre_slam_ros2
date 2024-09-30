cmake_minimum_required(VERSION 3.5)
project(dre_slam)

## Use C++14 as ROS2 supports it
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

## Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)

## Include directories
include_directories(
  ${rclcpp_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/include/
  ${PROJECT_SOURCE_DIR}/third_party/
)

## Declare a C++ library
add_library(${PROJECT_NAME}_lib
  core/src/dre_slam.cpp
  core/src/config.cpp
  core/src/ORBextractor.cpp
  core/src/camera.cpp
  core/src/tracking.cpp
  core/src/local_mapping.cpp
  core/src/loop_closing.cpp
  core/src/map.cpp
  core/src/map_point.cpp
  core/src/feature_detector.cpp
  core/src/frame.cpp
  core/src/keyframe.cpp
  core/src/encoder_integration.cpp
  core/src/common.cpp
  core/src/optimizer.cpp
  core/src/dynamic_pixel_detector.cpp
  core/src/dynamic_pixel_culling.cpp
  core/src/ros_puber.cpp
  core/src/octomap_fusion.cpp
  core/src/sub_octomap.cpp
  core/src/sub_octomap_construction.cpp
)

target_link_libraries(${PROJECT_NAME}_lib
  ${rclcpp_LIBRARIES}
	${OpenCV_LIBS}
	${EIGEN3_LIBS}
	${PCL_LIBRARIES}
	${CERES_LIBRARIES}
	${Sophus_LIBRARIES}
	${PROJECT_SOURCE_DIR}/third_party/DBoW2/lib/libDBoW2.so
	${PROJECT_SOURCE_DIR}/object_detector/lib/libobject_detector.so
	${OCTOMAP_LIBRARIES}
)

## Declare an executable
add_executable(dre_slam_node
  node/dre_slam_node.cpp)

target_link_libraries(dre_slam_node
  ${PROJECT_NAME}_lib
  ${rclcpp_LIBRARIES}
)

## Install targets
install(TARGETS dre_slam_node ${PROJECT_NAME}_lib
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY core/include/
  DESTINATION include/
)

## Ament export dependencies
ament_export_dependencies(rclcpp cv_bridge image_transport geometry_msgs message_filters pcl_ros)
ament_package()
