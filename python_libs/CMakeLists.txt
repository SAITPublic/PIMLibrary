if(TF_BUILD)
    add_subdirectory(tf_pim_ops)
endif()

if(PYTORCH_BUILD)
    add_subdirectory(py_pim_ops)
endif()
