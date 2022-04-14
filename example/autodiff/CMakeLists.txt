cmake_minimum_required(VERSION 3.1)
project(bfgs-example-autodiff)

if (NOT CMAKE_BUILD_TYPE)
	message(STATUS "No build type selected, default to Release")
	set(CMAKE_BUILD_TYPE "Release")
endif()

# set(CMAKE_CXX_STANDARD 14)

add_executable(${PROJECT_NAME} main.cpp)

include_directories("../../include")