aux_source_directory(. crfcode_source)
aux_source_directory(source crfcode_source)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wno-unused-command-line-argument")
set(CMAKE_C_FLAGS "-std=c11")

include_directories(./include)
add_executable(crfcodegen ${crfcode_source})

target_link_libraries (crfcodegen FimRuntime)
