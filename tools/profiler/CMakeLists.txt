aux_source_directory(. profiler_source)
add_executable(profiler ${profiler_source})

if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
target_link_libraries (profiler FimRuntime)
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
target_link_libraries (profiler FimRuntimeDev)
endif()
