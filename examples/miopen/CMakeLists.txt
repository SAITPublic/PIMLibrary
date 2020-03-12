cmake_minimum_required( VERSION 2.8.12 )

set(CMAKE_CXX_COMPILER "/opt/rocm/bin/hipcc")
set(HCC_CXX_FLAGS  "-hc --amdgpu-target=gfx900 --amdgpu-target=gfx906 --amdgpu-target=gfx908 ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HCC_CXX_FLAGS}")


add_executable(elt_wise_add_miopen eltwise_add_miopen.cpp)
add_executable(elt_wise_add_miopen_fimlib eltwise_add_miopen_fim.cpp)
include_directories(/opt/rocm/include)
target_link_libraries(elt_wise_add_miopen MIOpen FimRuntime)
target_link_libraries(elt_wise_add_miopen_fimlib MIOpen FimRuntime)

link_libraries(-amdgpu-target=gfx900 -Wno-unused-command-line-argument)
