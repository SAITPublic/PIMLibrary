set(CMAKE_CXX_COMPILER "/opt/rocm/bin/hipcc")
set(HCC_CXX_FLAGS  " --amdgpu-target=gfx906 --amdgpu-target=gfx908")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HCC_CXX_FLAGS}")

set(SOURCE0 fill_test.cpp)
#set(SOURCE1 multi_fill_test.cpp)
file(GLOB CRF_GEN_SOURCES ../../../../tools/crfcodegen/source/*.cpp)
cmake_minimum_required(VERSION 3.5.1)

include_directories(/opt/rocm/include)
include_directories(../../../../tools/crfcodegen/include)
link_directories(/opt/rocm/lib)

add_executable(fill_test.out ${SOURCE0} ${CRF_GEN_SOURCES})
#add_executable(multi_fill_test.out ${SOURCE1} ${CRF_GEN_SOURCES})

target_link_libraries(fill_test.out hsakmt)
#target_link_libraries(multi_fill_test.out hsakmt)

