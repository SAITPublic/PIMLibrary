if(NVIDIA AND OPENCL)
    set(CMAKE_CXX_COMPILER "g++")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    include_directories(${PIM_PATH}/include)
    link_directories(opencl/lib)
    link_directories(opencl/lib/x86_64/)
endif()

if(AMD)
    set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")
    set(HCC_CXX_FLAGS  " --amdgpu-target=gfx900 --amdgpu-target=gfx906 --amdgpu-target=gfx908")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HCC_CXX_FLAGS}")
    include_directories(${ROCM_PATH}/include)
    include_directories(${ROCM_PATH}/lib)
    include_directories(${ROCM_PATH}/opencl/include)
    link_directories(${ROCM_PATH}/opencl/lib)
    link_directories(${ROCM_PATH}/opencl/lib/x86_64/)
endif()

include_directories(${ROCM_PATH}/include)
include_directories(${ROCM_PATH}/opencl/include)
link_directories(${ROCM_PATH}/opencl/lib)
link_directories(${ROCM_PATH}/opencl/lib/x86_64/)

#file(GLOB HIP_SOURCES hip/*.cpp)
#add_executable(PimCompilerIntegrationTests ${HIP_SOURCES})
#target_link_libraries(PimCompilerIntegrationTests gtest PimRuntime PimC pimcompiler pimcfrontend codegen)

file(GLOB HIP_SOURCES_ELT hip/eltwise.cpp  hip/pimctest_main.cpp)
add_executable(PimCompilerIntegrationTestElt ${HIP_SOURCES_ELT})
target_link_libraries(PimCompilerIntegrationTestElt gtest PimRuntime PimC pimcompiler pimcfrontend codegen)

file(GLOB HIP_SOURCES_ELT_MULTIOP hip/elt_multiop.cpp  hip/pimctest_main.cpp)
add_executable(PimCompilerIntegrationTestEltMultiop ${HIP_SOURCES_ELT_MULTIOP})
target_link_libraries(PimCompilerIntegrationTestEltMultiop gtest PimRuntime PimC pimcompiler pimcfrontend codegen)

file(GLOB HIP_SOURCES_MAD hip/mad.cpp  hip/pimctest_main.cpp)
add_executable(PimCompilerIntegrationTestMad ${HIP_SOURCES_MAD})
target_link_libraries(PimCompilerIntegrationTestMad gtest PimRuntime PimC pimcompiler pimcfrontend codegen)

file(GLOB BLAS_HIP_SOURCES_AXPY hip/blas1/blas_axpy.cpp hip/pimctest_main.cpp)
add_executable(PimCompilerIntegrationTestAxpy ${BLAS_HIP_SOURCES_AXPY})
target_link_libraries(PimCompilerIntegrationTestAxpy gtest PimRuntime PimC pimcompiler pimcfrontend codegen)

file(GLOB BLAS_HIP_SOURCES_SCAL hip/blas1/blas_scal.cpp hip/pimctest_main.cpp)
add_executable(PimCompilerIntegrationTestScal ${BLAS_HIP_SOURCES_SCAL})
target_link_libraries(PimCompilerIntegrationTestScal gtest PimRuntime PimC pimcompiler pimcfrontend codegen)

# GEMV test cases
file(GLOB HIP_SOURCES_GEMV hip/gemm/gemv.cpp hip/pimctest_main.cpp)
add_executable(PimCompilerIntegrationTestGemv ${HIP_SOURCES_GEMV})
target_link_libraries(PimCompilerIntegrationTestGemv gtest PimRuntime PimC pimcompiler pimcfrontend codegen)

file(GLOB HIP_SOURCES_MULTI_INPUT_GEMV hip/gemm/multi_input_tile_gemv.cpp hip/pimctest_main.cpp)
add_executable(PimCompilerIntegrationTestMultiInputTileGemv ${HIP_SOURCES_MULTI_INPUT_GEMV})
target_link_libraries(PimCompilerIntegrationTestMultiInputTileGemv gtest PimRuntime PimC pimcompiler pimcfrontend codegen)

file(GLOB HIP_SOURCES_MULTI_OUTPUT_GEMV hip/gemm/multi_output_tile_gemv.cpp hip/pimctest_main.cpp)
add_executable(PimCompilerIntegrationTestMultiOutputTileGemv ${HIP_SOURCES_MULTI_OUTPUT_GEMV})
target_link_libraries(PimCompilerIntegrationTestMultiOutputTileGemv gtest PimRuntime PimC pimcompiler pimcfrontend codegen)

file(GLOB HIP_SOURCES_BATCH_GEMV hip/gemm/batch_gemv.cpp hip/pimctest_main.cpp)
add_executable(PimCompilerIntegrationTestBatchGemv ${HIP_SOURCES_BATCH_GEMV})
target_link_libraries(PimCompilerIntegrationTestBatchGemv gtest PimRuntime PimC pimcompiler pimcfrontend codegen)
