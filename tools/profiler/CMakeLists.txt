aux_source_directory(. profiler_source)
add_executable(profiler ${profiler_source})

target_link_libraries (profiler PimRuntime)

#target_link_libraries (profiler PimRuntime hip_hcc)
