aux_source_directory(. runtime_source)
aux_source_directory(manager  runtime_source)
aux_source_directory(executor runtime_source)
aux_source_directory(emulator runtime_source)
aux_source_directory(utility  runtime_source)

set(CMAKE_CXX_COMPILER "${ROCM_PATH}/hip/bin/hipcc")
set(HCC_CXX_FLAGS  "-hc --amdgpu-target=gfx803 --amdgpu-target=gfx900 --amdgpu-target=gfx906 --amdgpu-target=gfx908 ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HCC_CXX_FLAGS}")

add_library(FimRuntime SHARED ${runtime_source})
target_link_libraries (FimRuntime glog)
