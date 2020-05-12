from bokeh.io import show, output_file
from bokeh.models import Panel, Tabs

from visualizer.TableViz import create_table
from visualizer.TimelineViz import create
from visualizer.MainPage import create_main
from parser.ArgParser import arg_parser
from parser.FileParser import parse_csv_file, parse_fim_log_file, parse_miopen_log_file
from analyser.TimelineAnalyser import get_start_end_times
from analyser.TableAnalyser import get_table_stats

if __name__=='__main__':

	#Parse the Arguments
	args = arg_parser()

	#GPU File Visualization
	#Read File
	df_gpu=parse_csv_file(args.gpu_file)
	#Produce timeline plot
	event_names, start_times, end_times = get_start_end_times(df_gpu)
	timeline_plot = create(event_names, start_times, end_times, title = 'Timeline Plot', plot_height = 500, plot_width=1200, x_axis_label = 'Time (in ns)', y_axis_label = 'GPU Kernels')
	#Produce Tabular plot
	df_gpu_table = get_table_stats(df_gpu)
	table_plot = create_table(df_gpu_table, heading = 'GPU Calls Summary')
	#Create tabs for GPU calls Plots
	tab_gpu_timeline = Panel(child=timeline_plot, title="GPU Timeline Plot")
	tab_gpu_table = Panel(child=table_plot, title="GPU Calls Stats")
	tabs_gpu = Tabs(tabs=[tab_gpu_timeline, tab_gpu_table])

	#FIM Log File Visualization
	#Read File
	df_cpu,df_cpu_buf=parse_fim_log_file(args.fim_file)
	#Produce timeline plot
	event_names, start_times, end_times = get_start_end_times(df_cpu, 'ModuleName', 'BeginTime_us', 'EndTime_us')
	timeline_plot = create(event_names, start_times, end_times, title = 'Timeline Plot', plot_height = 500, plot_width=1200, x_axis_label = 'Time (in us)', y_axis_label = 'FIM Modules')
	#Produce Tabular plot for Module
	df_cpu_module = get_table_stats(df_cpu, 'ModuleName', 'Duration_us', avg_col = 'Average_us', total_col = 'TotalDuration_us')
	table_plot_m = create_table(df_cpu_module, heading = 'Module Summary')
	#Produce Tabular plot for APIs
	df_cpu_api = get_table_stats(df_cpu, 'APIName', 'Duration_us', avg_col = 'Average_us', total_col = 'TotalDuration_us')
	table_plot_a = create_table(df_cpu_api, heading = 'API Summary')
	#Produce Tabular plot for Buffers
	table_plot_b = create_table(df_cpu_buf, heading = 'FIM Buffers Summary')
	#Create tabs for CPU calls Plots
	tab_fim_timeline = Panel(child=timeline_plot, title="FIM Timeline Plot")
	tab_fim_module = Panel(child=table_plot_m, title="FIM Module Calls Stats")
	tab_fim_api = Panel(child=table_plot_a, title="FIM API Calls Stats")
	tab_fim_buffer = Panel(child=table_plot_b, title="FIM Buffer Calls")
	tabs_fim = Tabs(tabs=[tab_fim_timeline, tab_fim_module, tab_fim_api, tab_fim_buffer])

	#MIOpen Log File Visualization
	#Read File
	df_mi=parse_miopen_log_file(args.miopen_file)
	#Produce Tabular plot
	table_plot = create_table(df_mi, heading = 'MIOpen API Calls')
	#Create tabs for MIOpen Plot
	tab_mi_table = Panel(child=table_plot, title="MIOpen Function Calls")
	tabs_mi = Tabs(tabs=[tab_mi_table])

	#Main Page
	output_file(filename=args.output, title='Profiler Visualization', mode='inline')
	#Get the main page
	tabs = {'GPU Profile Data': tabs_gpu, 'FIM SDK Profile Data': tabs_fim, 'MIOpen APIs Profile Data': tabs_mi}
	main_plot = create_main(tabs)
	#Output Main Page
	show(main_plot)
