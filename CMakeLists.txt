cmake_minimum_required(VERSION 3.17)
project(library)

set(CMAKE_CXX_STANDARD 14)


#include_directories(alphazero/mcts)

#add_library(mcts alphazero/mcts/mcts.cpp)


#set(CMAKE_SYSTEM_PREFIX_PATH "/usr/local")
#set(CMAKE_SYSTEM_PREFIX_PATH  "/Users/sumeetbatra/opt/miniconda3/envs/RL/")

add_executable(main alphazero/mcts/main.cpp)


find_package(xtensor REQUIRED)
find_package(pybind11 REQUIRED)
#pybind11_add_module(test alphazero/mcts/test.cpp)

#target_include_directories(mcts PUBLIC ${xtensor_INCLUDE_DIRS})
#target_include_directories(main PUBLIC ${pybind11_INCLUDE_DIRS})
#target_link_libraries(library PUBLIC xtensor pybind11)
#target_link_libraries(main PUBLIC pybind11)
target_include_directories(main PUBLIC
        "${PROJECT_BINARY_DIR}"
        "${PROJECT_SOURCE_DIR}"
        ${pybind11_INCLUDE_DIRS}
        ${xtensor_INCLUDE_DIRS}
        )