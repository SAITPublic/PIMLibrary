aux_source_directory(. runtime_source)
add_library(FimRuntime SHARED ${runtime_source})
target_link_libraries (FimRuntime)
