set(CMAKE_CXX_COMPILER "/opt/rocm/bin/hipcc")
set(HCC_CXX_FLAGS  "--amdgpu-target=gfx906 --amdgpu-target=gfx908")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HCC_CXX_FLAGS}")

set(SOURCE mem_test.cpp)

cmake_minimum_required(VERSION 3.5.1)

include_directories(/opt/rocm/include)
link_directories(/opt/rocm/lib)

add_executable(mem_test.out ${SOURCE})

target_link_libraries(mem_test.out hsakmt)

