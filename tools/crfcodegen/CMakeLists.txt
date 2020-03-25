aux_source_directory(. crfcode_source)
add_executable(crfcodegen ${crfcode_source})

if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
target_link_libraries (crfcodegen FimRuntime)
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
target_link_libraries (crfcodegen FimRuntimeDev)
endif()
