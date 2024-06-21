cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(SubgraphMatching LANGUAGES CUDA CXX)
set(CMAKE_VERBOSE_MAKEFILE on)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

include_directories(/usr/local/cuda/include)

set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")  
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")  

add_library(graph_objects SHARED graph.cu graph.h graph_gpu.h)
add_library(candidate_objects SHARED candidate.cu candidate.h)
add_library(join_objects SHARED join_bfs.cu join_bfs.h join.cu join.h)

set_target_properties(graph_objects PROPERTIES
    CUDA_ARCHITECTURES "70"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

set_target_properties(candidate_objects PROPERTIES
    CUDA_ARCHITECTURES "70"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

set_target_properties(join_objects PROPERTIES
    CUDA_ARCHITECTURES "70"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(SubgraphMatching main.cu)

target_link_libraries(SubgraphMatching PUBLIC graph_objects candidate_objects join_objects ${CUDA_LIBRARIES})