set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")
set(HCC_CXX_FLAGS  " --amdgpu-target=gfx900 --amdgpu-target=gfx906 --amdgpu-target=gfx908")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HCC_CXX_FLAGS}")

include_directories(${ROCM_PATH}/include)
include_directories(${ROCM_PATH}/opencl/include)
link_directories(${ROCM_PATH}/opencl/lib)
link_directories(${ROCM_PATH}/opencl/lib/x86_64/)

file(GLOB HIP_SOURCES hip/*.cpp)
file(GLOB OPENCL_SOURCES opencl/*.cpp)

# add_subdirectory(monitor/opencl_sample)
add_executable(PimIntegrationTests ${HIP_SOURCES})
add_executable(OpenCLPimIntegrationTests ${OPENCL_SOURCES})

target_link_libraries(PimIntegrationTests gtest PimRuntime)
target_link_libraries(OpenCLPimIntegrationTests gtest_main gtest PimRuntime OpenCL)

