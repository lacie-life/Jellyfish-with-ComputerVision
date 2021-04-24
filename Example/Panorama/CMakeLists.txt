cmake_minimum_required(VERSION 3.19)

project(Panorama)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

#include_directories(include)

# Add the application sources to the target
add_executable(${PROJECT_NAME} main.cpp)

# Link librealsense2 to the target
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})