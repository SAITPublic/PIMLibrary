Profiler

Profiler is used to generate visualizations. It uses parser, analyser and visualizer modules for the same.


Requirements-
The modules depend on bokeh and pandas python library for generating visualizations. To install use-
```
pip install bokeh
pip intsall pandas
```

Testing - 
To run Profiler using sample test files in test folder and generate timeline and summarized tabular visualizations, directly run bokeh server:
```
bokeh serve --show .
```

To execute FimProfiler for any other file, provide name of the files with path in the -g, -f and -m flag after --args flag:
```
bokeh serve --show . --args -g <name_of_gpu_kernels_csv_file_with_path> -f <name_of_fim_log_file_with_path> -m <name_of_miopen_log_file_with_path>
```

Use -h flag to view help with all command line arguments.
```
bokeh serve --show . --args -h
```
