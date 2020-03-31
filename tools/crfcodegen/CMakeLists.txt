aux_source_directory(. crfcode_source)
add_executable(crfcodegen ${crfcode_source})

target_link_libraries (crfcodegen FimRuntime)
