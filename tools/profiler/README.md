Profiler

FimProfiler.py API is used to generate visualizations. It uses parser and visualizer modules for the same.


Requirements-
The modules depend on bokeh and pandas python library for generating visualizations. To install use-
```
pip install bokeh
pip intsall pandas
```

Testing - 
Execute FimProfiler using sample stat file in test folder and generate tabular visualization by directly calling it:
```
python FimProfiler.py
```

To execute FimProfiler for any other file, provide name of the file with path as the first argument:
```
python FimProfiler.py <name_of_csv_file_with_path>
```
