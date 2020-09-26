aux_source_directory(. crfcode_source)
aux_source_directory(source crfcode_source)

set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -Wno-unused-command-line-argument")
set(CMAKE_C_FLAGS "-std=c14")

include_directories(./include)
add_executable(crfcodegen ${crfcode_source})

target_link_libraries (crfcodegen FimRuntime hip_hcc)
