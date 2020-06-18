cmake_minimum_required(VERSION 2.8.3)
project(nn_adaptive_controller)
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  sensor_msgs
  std_msgs
  tf
  tf2
)

set(Caffe2_DIR "$ENV{HOME}/pytorch/torch/share/cmake/Caffe2")
set(Torch_DIR "$ENV{HOME}/pytorch/torch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if (NOT Torch_FOUND)
    message(FATAL_ERROR "libtorch not found, wavehands_stop will not built.")
else ()
    message(STATUS "++++ found libtorch ")
endif ()

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES nn_adaptive_controller
 CATKIN_DEPENDS geometry_msgs nav_msgs roscpp sensor_msgs std_msgs tf tf2
 DEPENDS
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

#############
# LIBRARIES #
#############

add_library(nn_adaptive_controller
    src/controller_network/architecture.cpp
    src/controller_network/custom_dataset.cpp)

target_link_libraries(nn_adaptive_controller ${catkin_LIBRARIES} ${TORCH_LIBRARIES})
set_property(TARGET nn_adaptive_controller PROPERTY CXX_STANDARD 14)

#########
# TESTS #
#########

catkin_add_gtest(test_architecture 
  test/test_architecture.cpp
)
target_link_libraries(test_architecture nn_adaptive_controller ${catkin_LIBRARIES} ${TORCH_LIBRARIES})
set_property(TARGET test_architecture PROPERTY CXX_STANDARD 14)
