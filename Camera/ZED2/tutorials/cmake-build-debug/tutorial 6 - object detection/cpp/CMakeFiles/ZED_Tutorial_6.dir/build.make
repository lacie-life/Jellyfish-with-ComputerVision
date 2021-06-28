# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/lacie/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/211.7442.42/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/lacie/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/211.7442.42/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lacie/Github/dip-learning/Camera/ZED2/tutorials

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lacie/Github/dip-learning/Camera/ZED2/tutorials/cmake-build-debug

# Include any dependencies generated for this target.
include tutorial\ 6\ -\ object\ detection/cpp/CMakeFiles/ZED_Tutorial_6.dir/depend.make

# Include the progress variables for this target.
include tutorial\ 6\ -\ object\ detection/cpp/CMakeFiles/ZED_Tutorial_6.dir/progress.make

# Include the compile flags for this target's objects.
include tutorial\ 6\ -\ object\ detection/cpp/CMakeFiles/ZED_Tutorial_6.dir/flags.make

tutorial\ 6\ -\ object\ detection/cpp/CMakeFiles/ZED_Tutorial_6.dir/main.o: tutorial\ 6\ -\ object\ detection/cpp/CMakeFiles/ZED_Tutorial_6.dir/flags.make
tutorial\ 6\ -\ object\ detection/cpp/CMakeFiles/ZED_Tutorial_6.dir/main.o: ../tutorial\ 6\ -\ object\ detection/cpp/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lacie/Github/dip-learning/Camera/ZED2/tutorials/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tutorial 6 - object detection/cpp/CMakeFiles/ZED_Tutorial_6.dir/main.o"
	cd "/home/lacie/Github/dip-learning/Camera/ZED2/tutorials/cmake-build-debug/tutorial 6 - object detection/cpp" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ZED_Tutorial_6.dir/main.o -c "/home/lacie/Github/dip-learning/Camera/ZED2/tutorials/tutorial 6 - object detection/cpp/main.cpp"

tutorial\ 6\ -\ object\ detection/cpp/CMakeFiles/ZED_Tutorial_6.dir/main.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ZED_Tutorial_6.dir/main.i"
	cd "/home/lacie/Github/dip-learning/Camera/ZED2/tutorials/cmake-build-debug/tutorial 6 - object detection/cpp" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/lacie/Github/dip-learning/Camera/ZED2/tutorials/tutorial 6 - object detection/cpp/main.cpp" > CMakeFiles/ZED_Tutorial_6.dir/main.i

tutorial\ 6\ -\ object\ detection/cpp/CMakeFiles/ZED_Tutorial_6.dir/main.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ZED_Tutorial_6.dir/main.s"
	cd "/home/lacie/Github/dip-learning/Camera/ZED2/tutorials/cmake-build-debug/tutorial 6 - object detection/cpp" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/lacie/Github/dip-learning/Camera/ZED2/tutorials/tutorial 6 - object detection/cpp/main.cpp" -o CMakeFiles/ZED_Tutorial_6.dir/main.s

# Object files for target ZED_Tutorial_6
ZED_Tutorial_6_OBJECTS = \
"CMakeFiles/ZED_Tutorial_6.dir/main.o"

# External object files for target ZED_Tutorial_6
ZED_Tutorial_6_EXTERNAL_OBJECTS =

tutorial\ 6\ -\ object\ detection/cpp/ZED_Tutorial_6: tutorial\ 6\ -\ object\ detection/cpp/CMakeFiles/ZED_Tutorial_6.dir/main.o
tutorial\ 6\ -\ object\ detection/cpp/ZED_Tutorial_6: tutorial\ 6\ -\ object\ detection/cpp/CMakeFiles/ZED_Tutorial_6.dir/build.make
tutorial\ 6\ -\ object\ detection/cpp/ZED_Tutorial_6: /usr/local/zed/lib/libsl_zed.so
tutorial\ 6\ -\ object\ detection/cpp/ZED_Tutorial_6: /usr/lib/x86_64-linux-gnu/libopenblas.so
tutorial\ 6\ -\ object\ detection/cpp/ZED_Tutorial_6: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
tutorial\ 6\ -\ object\ detection/cpp/ZED_Tutorial_6: /usr/lib/x86_64-linux-gnu/libcuda.so
tutorial\ 6\ -\ object\ detection/cpp/ZED_Tutorial_6: /usr/local/cuda/lib64/libcudart.so
tutorial\ 6\ -\ object\ detection/cpp/ZED_Tutorial_6: tutorial\ 6\ -\ object\ detection/cpp/CMakeFiles/ZED_Tutorial_6.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lacie/Github/dip-learning/Camera/ZED2/tutorials/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ZED_Tutorial_6"
	cd "/home/lacie/Github/dip-learning/Camera/ZED2/tutorials/cmake-build-debug/tutorial 6 - object detection/cpp" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ZED_Tutorial_6.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tutorial\ 6\ -\ object\ detection/cpp/CMakeFiles/ZED_Tutorial_6.dir/build: tutorial\ 6\ -\ object\ detection/cpp/ZED_Tutorial_6

.PHONY : tutorial\ 6\ -\ object\ detection/cpp/CMakeFiles/ZED_Tutorial_6.dir/build

tutorial\ 6\ -\ object\ detection/cpp/CMakeFiles/ZED_Tutorial_6.dir/clean:
	cd "/home/lacie/Github/dip-learning/Camera/ZED2/tutorials/cmake-build-debug/tutorial 6 - object detection/cpp" && $(CMAKE_COMMAND) -P CMakeFiles/ZED_Tutorial_6.dir/cmake_clean.cmake
.PHONY : tutorial\ 6\ -\ object\ detection/cpp/CMakeFiles/ZED_Tutorial_6.dir/clean

tutorial\ 6\ -\ object\ detection/cpp/CMakeFiles/ZED_Tutorial_6.dir/depend:
	cd /home/lacie/Github/dip-learning/Camera/ZED2/tutorials/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lacie/Github/dip-learning/Camera/ZED2/tutorials "/home/lacie/Github/dip-learning/Camera/ZED2/tutorials/tutorial 6 - object detection/cpp" /home/lacie/Github/dip-learning/Camera/ZED2/tutorials/cmake-build-debug "/home/lacie/Github/dip-learning/Camera/ZED2/tutorials/cmake-build-debug/tutorial 6 - object detection/cpp" "/home/lacie/Github/dip-learning/Camera/ZED2/tutorials/cmake-build-debug/tutorial 6 - object detection/cpp/CMakeFiles/ZED_Tutorial_6.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : tutorial\ 6\ -\ object\ detection/cpp/CMakeFiles/ZED_Tutorial_6.dir/depend

