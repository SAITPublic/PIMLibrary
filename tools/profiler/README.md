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

This shall start the server and profiler output will be displayed in the browser in the same machine. If the output is to be seen on a different machine, use the following argument while starting the server: 
```
bokeh serve --show . --allow-websocket-origin=<IP address of system>:5006 
```

The profiler output can then been seen from any other machine by going to the following address: 
```
http://<IP address of system where server is running>:5006 
```

To execute FimProfiler for any other file, provide name of the files with path in the -g, -f and -m flag after --args flag:
```
bokeh serve --show . --args -g <name_of_gpu_kernels_csv_file_with_path> -f <name_of_fim_log_file_with_path> -m <name_of_miopen_log_file_with_path>
```

Use -h flag to view help with all command line arguments.
```
bokeh serve --show . --args -h
```
