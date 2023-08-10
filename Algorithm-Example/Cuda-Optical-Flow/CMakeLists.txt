cmake_minimum_required(VERSION 3.17)
project(Cuda_Optical_Flow)

find_package( OpenCV REQUIRED )

set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})