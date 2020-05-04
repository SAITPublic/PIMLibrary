Profiler

FimProfiler.py API is used to generate visualizations. It uses parser, analyser and visualizer modules for the same.


Requirements-
The modules depend on bokeh and pandas python library for generating visualizations. To install use-
```
pip install bokeh
pip intsall pandas
```

Testing - 
To execute FimProfiler using sample csv file(containing GPU Kernel Calls) and sample log file (containing CPU API calls) in test folder and generate timeline and summarized tabular visualizations, directly call it:
```
python FimProfiler.py
```

To execute FimProfiler for any other file, provide name of the files with path in the -g and -c flag and to provide the output file name use the -o flag:
```
python FimProfiler.py -g <name_of_csv_file_with_path> -c <name_of_log_file_with_path> -o <output_file_nam>
```

Use -h flag to view help with all command line arguments.
```
python FimProfiler.py -h
```
