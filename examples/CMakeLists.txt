add_subdirectory(integration_test)

if(AMD)
add_subdirectory(unit_test)
endif()

if(PIM_COMPILER)
    add_subdirectory(pim_compiler)
endif()
