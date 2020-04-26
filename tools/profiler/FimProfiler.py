from bokeh.io import show, output_file
from bokeh.layouts import column

from visualizer.TableViz import create_table
from visualizer.TimelineViz import create
from parser.ArgParser import arg_parser
from parser.FileParser import parse_csv_file, parse_log_file
from analyser.TimelineAnalyser import get_start_end_times
from analyser.TableAnalyser import get_table_stats

if __name__=='__main__':

	#Parse the Arguments
	args = arg_parser()

	#GPU File Visualization
	output_file(filename=args.gpu_output, title='GPU Visualization', mode='inline')
	#Read File
	df_gpu=parse_csv_file(args.gpu_file)
	#Produce timeline plot
	event_names, start_times, end_times = get_start_end_times(df_gpu)
	timeline_plot = create(event_names, start_times, end_times, title = 'Timeline Plot', plot_height = 500, plot_width=1200)
	#Produce Tabular plot
	df_gpu_table = get_table_stats(df_gpu)
	heading, table_plot = create_table(df_gpu_table, heading = 'GPU Calls Summary')
	#Output all GPU calls Plots
	show(column(timeline_plot,heading, table_plot))

	#CPU File Visualization
	output_file(filename=args.cpu_output, title='CPU Visualization', mode='inline')
	#Read File
	df_cpu=parse_log_file(args.cpu_file)
	#Produce timeline plot
	event_names, start_times, end_times = get_start_end_times(df_cpu, 'ModuleName', 'BeginTime_us', 'EndTime_us')
	timeline_plot = create(event_names, start_times, end_times, title = 'Timeline Plot', plot_height = 500, plot_width=1200)
	#Produce Tabular plot for Module
	df_cpu_module = get_table_stats(df_cpu, 'ModuleName', 'Duration_us', avg_col = 'Average_us', total_col = 'TotalDuration_us')
	heading_m, table_plot_m = create_table(df_cpu_module, heading = 'Module Summary')
	#Produce Tabular plot for APIs
	df_cpu_api = get_table_stats(df_cpu, 'APIName', 'Duration_us', avg_col = 'Average_us', total_col = 'TotalDuration_us')
	heading_a, table_plot_a = create_table(df_cpu_api, heading = 'API Summary')
	#Output all CPU calls Plots
	show(column(timeline_plot, heading_m, table_plot_m, heading_a, table_plot_a))
