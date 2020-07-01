cmake_minimum_required(VERSION 2.8.3)
project(nn_adaptive_controller)
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  mav_msgs
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
find_package(Eigen3 REQUIRED)
add_definitions(${EIGEN3_DEFINITIONS})

if (NOT Torch_FOUND)
    message(FATAL_ERROR "libtorch not found, wavehands_stop will not built.")
else ()
    message(STATUS "++++ found libtorch ")
endif ()

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES nn_adaptive_controller
 CATKIN_DEPENDS geometry_msgs mav_msgs nav_msgs roscpp sensor_msgs std_msgs tf tf2
 DEPENDS
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

#############
# LIBRARIES #
#############

add_library(neural_adaptive_controller
    # src/controller_network/architecture.cpp
    # src/controller_network/custom_dataset.cpp
    # src/controller_network/dataset_reader.cpp
    src/neural_adaptive_controller/neural_adaptive_controller.cpp)

add_library(nn_adaptive_controller
    src/controller_network/architecture.cpp
    src/controller_network/custom_dataset.cpp
    src/controller_network/dataset_reader.cpp)

target_link_libraries(neural_adaptive_controller ${catkin_LIBRARIES} ${TORCH_LIBRARIES})
set_property(TARGET neural_adaptive_controller PROPERTY CXX_STANDARD 14)

target_link_libraries(nn_adaptive_controller ${catkin_LIBRARIES} ${TORCH_LIBRARIES})
set_property(TARGET nn_adaptive_controller PROPERTY CXX_STANDARD 14)

add_executable(dataset_gen_node src/controller_network/dataset_gen_node.cpp)
add_dependencies(dataset_gen_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(dataset_gen_node nn_adaptive_controller ${catkin_LIBRARIES} ${TORCH_LIBRARIES})

add_executable(neural_adaptive_controller_node src/neural_adaptive_controller/neural_adaptive_controller_node.cpp)
add_dependencies(neural_adaptive_controller_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(neural_adaptive_controller_node neural_adaptive_controller nn_adaptive_controller ${catkin_LIBRARIES})
set_property(TARGET neural_adaptive_controller_node PROPERTY CXX_STANDARD 14)

#########
# TESTS #
#########

catkin_add_gtest(test_architecture 
  test/test_architecture.cpp
)
target_link_libraries(test_architecture nn_adaptive_controller ${catkin_LIBRARIES} ${TORCH_LIBRARIES})
set_property(TARGET test_architecture PROPERTY CXX_STANDARD 14)

# catkin_add_gtest(test_controller
#   test/test_controller.cpp
# )
# target_link_libraries(test_controller nn_adaptive_controller neural_adaptive_controller ${catkin_LIBRARIES} ${TORCH_LIBRARIES})
# set_property(TARGET test_controller PROPERTY CXX_STANDARD 14)
