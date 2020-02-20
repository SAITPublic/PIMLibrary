cmake_minimum_required( VERSION 2.8.12 )

add_executable(elt_wise_add eltwise_add_miopen.cpp)
include_directories(/opt/rocm/include)
target_link_libraries(elt_wise_add MIOpen)

link_libraries(-amdgpu-target=gfx900 -Wno-unused-command-line-argument)

# Cmake does not add flags correctly for gcc
install(TARGETS elt_wise_add
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    DESTINATION ${MIOPEN_INSTALL_DIR}/bin)
