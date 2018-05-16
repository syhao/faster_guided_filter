# Author: Sudeep Pillai (Sep 14, 2014)
project (fast-guided-filter)
cmake_minimum_required (VERSION 2.6.0)

# Include package config, boost-python
find_package(PkgConfig REQUIRED)
include(cmake/boost-python.cmake)

# configure opencv
pkg_check_modules(OpenCV opencv)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/lib/python2.7/dist-packages/numpy/core/include)

# Include python (use -D flags instead)
SET(PYTHON_INCLUDE_DIRS "/usr/include/python2.7")
SET(PYTHON_LIBRARIES "/usr/lib/x86_64-linux-gnu/libpython2.7.so")

# Build np<=>opencv converter library
boost_python_module(np_opencv_converter np_opencv_converter.cpp utils/conversion.cpp)
target_link_libraries(np_opencv_converter boost_system boost_python ${OpenCV_LDFLAGS})

# Build test library
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
boost_python_module(fast_guided_filter tests/fast_guided_filter.cpp)
target_link_libraries(fast_guided_filter boost_system boost_python np_opencv_converter ${OpenCV_LDFLAGS})
