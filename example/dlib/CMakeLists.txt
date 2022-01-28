cmake_minimum_required(VERSION 3.1)
project(bfgs-example-dlib)

if (NOT CMAKE_BUILD_TYPE)
	message(STATUS "No build type selected, default to Release")
	set(CMAKE_BUILD_TYPE "Release")
endif()

add_executable(${PROJECT_NAME} main.cpp)

include_directories("../../include")

# dlib
find_package(dlib REQUIRED)
include_directories(${DLib_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} dlib::dlib)