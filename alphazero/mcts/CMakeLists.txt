cmake_minimum_required(VERSION 3.17)
project(..)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(..
        mcts.cpp
        mcts.h)

add_library(. mcts.cpp)

set(CMAKE_SYSTEM_PREFIX_PATH "/usr/local")

find_package(xtensor REQUIRED)
target_include_directories(. PUBLIC ${xtensor_INCLUDE_DIRS})
target_link_libraries(. PUBLIC xtensor)
