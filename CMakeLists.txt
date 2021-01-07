cmake_minimum_required(VERSION 3.17)
project(library)

set(CMAKE_CXX_STANDARD 11)

include_directories(/usr/local/lib/pkgconfig)
include_directories(/Users/sumeetbatra/opt/miniconda3/envs/RL/include)

find_package(xtensor REQUIRED)
include_directories(${xtensor_INCLUDE_DIR})
add_library(mcts alphazero/mcts/mcts.cpp)
add_subdirectory(${PROJECT_SOURCE_DIR}/extern/thc-chess-library/ EXCLUDE_FROM_ALL)
include_directories(${THC_CHESS_INCLUDE})
target_link_libraries(mcts ${xtensor_LIBRARIES} thc_chess)

add_executable(main alphazero/mcts/main.cpp)
target_link_libraries(main mcts)


target_include_directories(main PUBLIC
        "${PROJECT_BINARY_DIR}"
        "${PROJECT_SOURCE_DIR}"
        ${xtensor_INCLUDE_DIRS}
        )