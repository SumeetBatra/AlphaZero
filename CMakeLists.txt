cmake_minimum_required(VERSION 3.17)
project(library)

set(CMAKE_CXX_STANDARD 14)

include_directories(/usr/local/lib/pkgconfig)

find_package(xtensor REQUIRED)
include_directories(${xtensor_INCLUDE_DIR})
add_executable(mcts alphazero/mcts/mcts.cpp)
target_link_libraries(mcts ${xtensor_LIBRARIES})

add_executable(main alphazero/mcts/main.cpp)


target_include_directories(main PUBLIC
        "${PROJECT_BINARY_DIR}"
        "${PROJECT_SOURCE_DIR}"
        ${xtensor_INCLUDE_DIRS}
        )