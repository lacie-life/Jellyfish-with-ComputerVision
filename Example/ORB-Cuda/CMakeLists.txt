cmake_minimum_required(VERSION 3.19)
project(ORB_Cuda)

set(CMAKE_CXX_STANDARD 14)

# FIND LIBRARIES
find_package(OpenCV REQUIRED)
find_package(CUDA 10.2 REQUIRED)

# INCLUDE HEADERS
include_directories(${OpenCV_INCLUDE_DIRS})

# COMPILER FLAGS AND DEFINITIONS
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=sm_50)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11)

cuda_add_executable(ORB_Cuda main_gpu.cpp)
cuda_add_executable(ORB_CPU main_cpu.cpp)

target_link_libraries(ORB_Cuda ${OpenCV_LIBS})
target_link_libraries(ORB_CPU ${OpenCV_LIBS})