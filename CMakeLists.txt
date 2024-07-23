cmake_minimum_required(VERSION 3.10)  
  
project(CudaProject VERSION 1.0 LANGUAGES CUDA CXX)  
  
set(CMAKE_CXX_STANDARD 17)  
set(CMAKE_CXX_STANDARD_REQUIRED True)  
  
find_package(CUDA REQUIRED)  
  
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++17")  
  
include_directories(.)  
  
cuda_add_executable(CudaProject  
    main.cu  
    candidate.cu  
    join_bfs.cu  
    graph.cu  
)  
  
set_target_properties(CudaProject PROPERTIES CUDA_SEPARABLE_COMPILATION ON)