set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")
set(HCC_CXX_FLAGS  " --amdgpu-target=gfx900 --amdgpu-target=gfx906 --amdgpu-target=gfx908")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HCC_CXX_FLAGS}")

include_directories(${ROCM_PATH}/include)
include_directories(${ROCM_PATH}/opencl/include)
link_directories(${ROCM_PATH}/opencl/lib)
link_directories(${ROCM_PATH}/opencl/lib/x86_64/)

#file(GLOB HIP_SOURCES hip/*.cpp)
#add_executable(PimCompilerIntegrationTests ${HIP_SOURCES})
#target_link_libraries(PimCompilerIntegrationTests gtest PimRuntime PimC pimcompiler codegen)

file(GLOB HIP_SOURCES_ELT hip/eltwise.cpp  hip/pimctest_main.cpp)
add_executable(PimCompilerIntegrationTestElt ${HIP_SOURCES_ELT})
target_link_libraries(PimCompilerIntegrationTestElt gtest PimRuntime PimC pimcompiler codegen)

file(GLOB HIP_SOURCES_ELT_MULTIOP hip/elt_multiop.cpp  hip/pimctest_main.cpp)
add_executable(PimCompilerIntegrationTestEltMultiop ${HIP_SOURCES_ELT_MULTIOP})
target_link_libraries(PimCompilerIntegrationTestEltMultiop gtest PimRuntime PimC pimcompiler codegen)

#file(GLOB HIP_SOURCES_GEMV hip/gemv.cpp hip/pimctest_main.cpp)
#add_executable(PimCompilerIntegrationTestGemv ${HIP_SOURCES_GEMV})
#target_link_libraries(PimCompilerIntegrationTestGemv gtest PimRuntime PimC pimcompiler codegen)
#
#file(GLOB HIP_SOURCES_GEMVBATCH hip/gemv_batch.cpp hip/pimctest_main.cpp)
#add_executable(PimCompilerIntegrationTestGemvBatch ${HIP_SOURCES_GEMVBATCH})
#target_link_libraries(PimCompilerIntegrationTestGemvBatch gtest PimRuntime PimC pimcompiler codegen)
#
#file(GLOB BLAS_HIP_SOURCES_AXPY hip/blas1/blas_axpy.cpp hip/pimctest_main.cpp)
#add_executable(PimCompilerIntegrationTestAxpy ${BLAS_HIP_SOURCES_AXPY})
#target_link_libraries(PimCompilerIntegrationTestAxpy gtest PimRuntime PimC pimcompiler codegen)
