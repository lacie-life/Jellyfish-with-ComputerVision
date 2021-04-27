cmake_minimum_required(VERSION 3.19)
project(NARF_Keypoint)

set(CMAKE_CXX_STANDARD 14)

set(PCL_INCLUDE_DIRS /usr/local/include/pcl-1.8)

find_package(PCL REQUIRED)
find_package (Eigen3 3.3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(NARF_Keypoint main.cpp)

target_link_libraries(NARF_Keypoint ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_FILTERS_LIBRARIES})
target_link_libraries (NARF_Keypoint ${PCL_LIBRARIES})
target_link_libraries (NARF_Keypoint ${EIGEN3_LIBRARIES})
