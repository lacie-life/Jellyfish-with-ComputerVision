cmake_minimum_required(VERSION 3.19)
project(StatisticalOutlierRemoval)

set(PCL_INCLUDE_DIRS /usr/local/include/pcl-1.8)
find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(StatisticalOutlierRemoval main.cpp)

target_link_libraries(StatisticalOutlierRemoval ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_FILTERS_LIBRARIES})
target_link_libraries (StatisticalOutlierRemoval ${PCL_LIBRARIES})