cmake_minimum_required(VERSION 3.17)
project(library)

#set(CMAKE_CXX_STANDARD 14)
#
#include_directories(/usr/local/lib/pkgconfig)
#include_directories(/Users/sumeetbatra/opt/miniconda3/envs/RL/include)
#include_directories(/Users/sumeetbatra/AlphaZero/extern/libtorch/include)
#include_directories(/Users/sumeetbatra/AlphaZero/extern/libtorch/include/torch/csrc/api/include)
#
#list(APPEND CMAKE_PREFIX_PATH /Users/sumeetbatra/AlphaZero/extern/libtorch/share/cmake/Torch)
#
#find_package(xtensor REQUIRED)
#find_package(Torch REQUIRED)
#message(STATUS "Find Torch VERSION: ${Torch_VERSION}")
#include_directories(${xtensor_INCLUDE_DIRS})
#include_directories(${Torch_INCLUDE_DIRS})
#add_library(mcts alphazero/mcts/mcts.cpp)
#add_subdirectory(${PROJECT_SOURCE_DIR}/extern/thc-chess-library/ EXCLUDE_FROM_ALL)
#include_directories(${THC_CHESS_INCLUDE})
#target_link_libraries(mcts ${xtensor_LIBRARIES}
#                            ${Torch_LIBRARIES}
#                            thc_chess)
#
#add_executable(main alphazero/mcts/main.cpp)
#target_link_libraries(main mcts)
#
#
#
#
#
#target_include_directories(main PUBLIC
#        "${PROJECT_BINARY_DIR}"
#        "${PROJECT_SOURCE_DIR}"
#        ${xtensor_INCLUDE_DIRS}
#        )

list(APPEND CMAKE_PREFIX_PATH /Users/sumeetbatra/libtorch)
find_package(Torch REQUIRED)

add_executable(test alphazero/mcts/main.cpp)
target_link_libraries(test "${TORCH_LIBRARIES}")
set_property(TARGET test PROPERTY CXX_STANDARD 14)