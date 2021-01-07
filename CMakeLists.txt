cmake_minimum_required(VERSION 3.17)
project(library)

set(CMAKE_CXX_STANDARD 14)


#include_directories(alphazero/mcts)

#add_library(mcts alphazero/mcts/mcts.cpp)


#set(CMAKE_SYSTEM_PREFIX_PATH "/usr/local")
#set(CMAKE_SYSTEM_PREFIX_PATH  "/Users/sumeetbatra/opt/miniconda3/envs/RL/")

set(SOURCE_FILES main.cpp)
add_executable(main alphazero/mcts/main.cpp)
add_library(mcts_lib alphazero/mcts/mcts.cpp)


find_package(xtensor REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
target_link_libraries(main mcts_lib xtensor)
target_link_libraries(main PUBLIC ${Boost_LIBRARIES})
target_include_directories(mcts_lib PUBLIC ${xtensor_INCLUDE_DIRS})

#add_library(chess_utils alphazero/mcts/chess_utils.cpp)
pybind11_add_module(chess_utils alphazero/mcts/chess_utils.cpp)
target_include_directories(chess_utils PUBLIC ${pybind11_INCLUDE_DIRS})


#target_include_directories(mcts PUBLIC ${xtensor_INCLUDE_DIRS})
#target_include_directories(main PUBLIC ${pybind11_INCLUDE_DIRS})
#target_link_libraries(library PUBLIC xtensor pybind11)
#target_link_libraries(main PUBLIC pybind11)
target_include_directories(main PUBLIC
        "${PROJECT_BINARY_DIR}"
        "${PROJECT_SOURCE_DIR}"
        ${pybind11_INCLUDE_DIRS}
        ${xtensor_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        )