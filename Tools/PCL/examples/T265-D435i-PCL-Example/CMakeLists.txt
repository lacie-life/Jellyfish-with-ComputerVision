cmake_minimum_required(VERSION 3.19)
project(T265_D435i_PCL_Example)

set(CMAKE_CXX_STANDARD 14)
set(PCL_INCLUDE_DIRS /usr/local/include/pcl-1.8)

# Find librealsense2 installed package
find_package(realsense2 REQUIRED)
find_package( OpenCV REQUIRED )
find_package(PCL REQUIRED)
find_package (Eigen3 3.3 REQUIRED)

set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include_directories(include)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
include_directories(${EIGEN3_INCLUDE_DIR})


add_executable(${PROJECT_NAME} main.cpp)

# Link librealsense2 to the target
target_link_libraries(${PROJECT_NAME} ${realsense2_LIBRARY} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_FILTERS_LIBRARIES})
target_link_libraries (${PROJECT_NAME} ${PCL_LIBRARIES})
target_link_libraries (${PROJECT_NAME} ${EIGEN3_LIBRARIES})