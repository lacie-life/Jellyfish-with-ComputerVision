cmake_minimum_required(VERSION 3.19)
project(ORB_feature)

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

find_package( OpenCV REQUIRED )
find_package(Eigen3 REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${CSPARSE_INCLUDE_DIR}
)

add_executable( ORB_feature main.cpp  )
target_link_libraries( ORB_feature ${OpenCV_LIBS} Eigen3::Eigen)