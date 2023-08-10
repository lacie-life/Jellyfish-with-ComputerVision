cmake_minimum_required(VERSION 3.19)
project(PlanarSegmentaion)

set(CMAKE_CXX_STANDARD 14)

set(PCL_INCLUDE_DIRS /usr/local/include/pcl-1.8)

find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(PlanarSegmentaion main.cpp)

target_link_libraries(PlanarSegmentaion ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_FILTERS_LIBRARIES})
target_link_libraries (PlanarSegmentaion ${PCL_LIBRARIES})