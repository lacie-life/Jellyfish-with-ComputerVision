###############################################################################
## project settings ###########################################################
###############################################################################
# CMake documentation: https://cmake.org/cmake/help/v3.2/
# CMake simple example: http://derekmolloy.ie/hello-world-introductions-to-cmake/
# CMake cuda example: https://codeyarns.com/2013/09/13/how-to-build-cuda-programs-using-cmake/

# this project is set up such that it compiles
project(Cuda-OpenCV-Example)

# compiler options, set c++11 (requires cmake 3.2 or higher)
cmake_minimum_required(VERSION 3.2)
set(CMAKE_CXX_STANDARD 11)

###############################################################################
## find libraries #############################################################
###############################################################################

# FIND LIBRARIES
find_package(OpenCV REQUIRED)
find_package(CUDA 10.2 REQUIRED)

# INCLUDE HEADERS
include_directories(${OpenCV_INCLUDE_DIRS})

###############################################################################
## file globbing ##############################################################
###############################################################################

# find source and header files.
set(hello "hello.cu")
set(add "add.cu")
set(bgrtogray "bgrtogray.cu")
set(invert_1 "invert_1.cu")
set(invert_2 "invert_2.cu")
set(ptp "ptp.cu")
set(diff_1 "diff_1.cu")
set(diff_2 "diff_2.cu")
set(conversions "conversions.cpp")
set(diff_proper "diff_proper.cu")
set(split "split.cu")

###############################################################################
## target definitions #########################################################
###############################################################################

# COMPILER FLAGS AND DEFINITIONS
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=sm_50)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11)

# SPECIFY TARGETS
cuda_add_executable(hello ${hello})
cuda_add_executable(add ${add})
cuda_add_executable(bgrtogray ${bgrtogray})
cuda_add_executable(invert_1 ${invert_1})
cuda_add_executable(invert_2 ${invert_2})
cuda_add_executable(ptp ${ptp})
cuda_add_executable(diff_1 ${diff_1})
cuda_add_executable(diff_2 ${diff_2})
cuda_add_executable(conversions ${conversions})
cuda_add_executable(diff_proper ${diff_proper})
cuda_add_executable(split ${split})

target_link_libraries(bgrtogray ${OpenCV_LIBS})
target_link_libraries(invert_1 ${OpenCV_LIBS})
target_link_libraries(invert_2 ${OpenCV_LIBS})
target_link_libraries(ptp ${OpenCV_LIBS})
target_link_libraries(diff_1 ${OpenCV_LIBS})
target_link_libraries(diff_2 ${OpenCV_LIBS})
target_link_libraries(conversions ${OpenCV_LIBS})
target_link_libraries(diff_proper ${OpenCV_LIBS})
target_link_libraries(split ${OpenCV_LIBS})

###############################################################################
## testing ####################################################################
###############################################################################



###############################################################################
## output #####################################################################
###############################################################################