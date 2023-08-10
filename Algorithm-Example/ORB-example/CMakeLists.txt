cmake_minimum_required(VERSION 3.19)
project(ORB_example)

set(CMAKE_CXX_STANDARD 14)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( ORB_example main.cpp
                            stats.h
                            utils.h
                            tracker.h tracker.cpp)
target_link_libraries( ORB_example ${OpenCV_LIBS} )