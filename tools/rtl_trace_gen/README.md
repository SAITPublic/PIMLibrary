## RTL Trace Generator

this is testing tool for rtl. this generates memory trace and simulation result.  

## How to use

1. Set ini file (system_hbm_vega20.ini)

    - MEM_TRACE_OUTPUT = true
    - MEM_TRACE_PATH = /tmp/pim_rtl/

        notice) Do not change path

2. build

    - ./scripts/build.sh all -o .


3. Execute python test file.

    - python3 rtl_gemv_test.py 

        notice) this file should be executed at current path. 





