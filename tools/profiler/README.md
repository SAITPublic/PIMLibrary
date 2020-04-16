Profiler

FimProfiler.py API is used to generate visualizations. It uses parser, analyser and visualizer modules for the same.


Requirements-
The modules depend on bokeh and pandas python library for generating visualizations. To install use-
```
pip install bokeh
pip intsall pandas
```

Testing - 
To execute FimProfiler using sample csv file(containing Kernel Calls) in test folder and generate timeline and summarized tabular visualization, directly call it:
```
python FimProfiler.py
```

To execute FimProfiler for any other file, provide name of the file with path in the -i flag and to provide the output file name use the -o flag:
```
python FimProfiler.py -i <name_of_csv_file_with_path> -o <output_file_name>
```

Use -h flag to view help with command line arguments.
```
python FimProfiler.py -h
```
