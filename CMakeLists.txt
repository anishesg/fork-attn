cmake_minimum_required(VERSION 3.18)
project(fork_attn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -O3")

include_directories(src)

add_library(fork_attn_lib STATIC
    src/naive_decode_attn.cu
    src/shared_prefix_attn.cu
    src/branch_suffix_attn.cu
    src/fork_attn.cu
    src/multi_level_fork_attn.cu
)
target_include_directories(fork_attn_lib PUBLIC src)
target_link_libraries(fork_attn_lib CUDA::cublas)

add_executable(test_correctness tests/test_correctness.cu)
target_link_libraries(test_correctness fork_attn_lib CUDA::cudart)

add_executable(bench_latency tests/bench_latency.cu)
target_link_libraries(bench_latency fork_attn_lib CUDA::cudart)

add_executable(test_multi_level tests/test_multi_level.cu)
target_link_libraries(test_multi_level fork_attn_lib CUDA::cudart)
