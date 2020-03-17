set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")
set(HCC_CXX_FLAGS  "-hc --amdgpu-target=gfx900 --amdgpu-target=gfx906 --amdgpu-target=gfx908 ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HCC_CXX_FLAGS}")

include_directories(/opt/rocm/include)
file(GLOB SOURCES hip/*.cpp)
add_executable(FimIntegrationTests ${SOURCES})

target_link_libraries(FimIntegrationTests dramsim2 gtest_main gtest FimRuntime)
add_subdirectory(miopen)
#add_subdirectory(tensorflow)
