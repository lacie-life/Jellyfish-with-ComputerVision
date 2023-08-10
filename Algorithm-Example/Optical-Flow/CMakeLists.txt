cmake_minimum_required(VERSION 3.17)
project(Optical_Flow)

set(CMAKE_CXX_STANDARD 14)
find_package( OpenCV REQUIRED )

set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
add_executable(${PROJECT_NAME} main.cpp)

# Link librealsense2 to the target
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})