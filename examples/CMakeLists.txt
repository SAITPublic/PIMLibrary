set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")
set(HCC_CXX_FLAGS  " --amdgpu-target=gfx906 --amdgpu-target=gfx908")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HCC_CXX_FLAGS}")

include_directories(${ROCM_PATH}/include)
file(GLOB HIP_SOURCES hip/*.cpp)

if(MIOPEN_APPS)
    file(GLOB MIOPEN_SOURCES miopen/*.cpp)
endif()

add_executable(PimIntegrationTests ${HIP_SOURCES} ${MIOPEN_SOURCES})

if(MIOPEN_APPS)
    target_link_libraries(PimIntegrationTests MIOpen gtest_main gtest PimRuntime)
else()
    target_link_libraries(PimIntegrationTests gtest_main gtest PimRuntime)
endif()
