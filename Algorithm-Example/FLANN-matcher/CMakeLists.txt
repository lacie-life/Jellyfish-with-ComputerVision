cmake_minimum_required(VERSION 3.19)
project(FLANN_matcher)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(CUDA 10.2 REQUIRED)

set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include_directories(${OpenCV_INCLUDE_DIRS})

#include_directories(include)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_53,code=sm_53)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11)

# Add the application sources to the target
cuda_add_executable(${PROJECT_NAME} main.cpp)

# Link librealsense2 to the target
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})