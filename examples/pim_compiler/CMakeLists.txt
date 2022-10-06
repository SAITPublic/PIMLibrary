set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")
set(HCC_CXX_FLAGS  " --amdgpu-target=gfx900 --amdgpu-target=gfx906 --amdgpu-target=gfx908")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HCC_CXX_FLAGS}")

include_directories(${ROCM_PATH}/include)
include_directories(${ROCM_PATH}/opencl/include)
link_directories(${ROCM_PATH}/opencl/lib)
link_directories(${ROCM_PATH}/opencl/lib/x86_64/)

file(GLOB HIP_SOURCES hip/*.cpp)
#file(GLOB BLAS_HIP_SOURCES hip/blas_1/*.cpp)

add_executable(PimCompilerIntegrationTests ${HIP_SOURCES})
#add_executable(PimCompilerIntegrationTests ${HIP_SOURCES} ${BLAS_HIP_SOURCES})

target_link_libraries(PimCompilerIntegrationTests gtest PimRuntime PimC pimcompiler codegen)
