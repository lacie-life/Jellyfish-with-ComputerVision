cmake_minimum_required(VERSION 3.19)
project(Voxel_Grid)

set(CMAKE_CXX_STANDARD 14)

set(PCL_INCLUDE_DIRS /usr/local/include/pcl-1.8)
find_package(PCL REQUIRED)
find_package (Eigen3 3.3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(Voxel_Grid main.cpp)

target_link_libraries(Voxel_Grid ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_FILTERS_LIBRARIES})
target_link_libraries (Voxel_Grid ${PCL_LIBRARIES})
target_link_libraries (Voxel_Grid ${EIGEN3_LIBRARIES})
