cmake_minimum_required(VERSION 3.10)

project(SubgraphMatching CUDA)

set(CMAKE_CUDA_STANDARD 14)

# Add source files.
set(SRC_FILES 
    candidate.cu 
    candidate.h 
    graph_gpu.h 
    graph.cu 
    graph.h
    helper.h 
    join_bfs.cu
    join_bfs.h
    join.h
    join.cu
    main.cu
)

# Add executable file.
add_executable(SubgraphMatching ${SRC_FILES})

set(CMAKE_BUILD_TYPE "Debug")