set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")
set(HCC_CXX_FLAGS  " --amdgpu-target=gfx906")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HCC_CXX_FLAGS}")

include_directories(/opt/rocm/include)
file(GLOB HIP_SOURCES hip/*.cpp)

if(MIOPEN_APPS)
    file(GLOB MIOPEN_SOURCES miopen/*.cpp)
endif()

add_executable(FimIntegrationTests ${HIP_SOURCES} ${MIOPEN_SOURCES})

if(MIOPEN_APPS)
    target_link_libraries(FimIntegrationTests MIOpen gtest_main gtest FimRuntime)
else()
    target_link_libraries(FimIntegrationTests gtest_main gtest FimRuntime)
endif()
