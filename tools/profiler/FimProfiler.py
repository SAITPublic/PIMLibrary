from bokeh.io import show, output_file, save
from bokeh.layouts import column

from visualizer.TableViz import create_table
from visualizer.TimelineViz import create
from visualizer.MainPage import create_main, get_content
from parser.ArgParser import arg_parser
from parser.FileParser import parse_csv_file, parse_fim_log_file, parse_miopen_log_file
from analyser.TimelineAnalyser import get_start_end_times
from analyser.TableAnalyser import get_table_stats

if __name__=='__main__':

	#Parse the Arguments
	args = arg_parser()
	output_name = args.output
	gpu_output = output_name[:-5] + '_gpu.html'
	cpu_output = output_name[:-5] + '_fim.html'
	mi_output = output_name[:-5] + '_mi.html'

	#GPU File Visualization
	output_file(filename=gpu_output, title='GPU Visualization', mode='inline')
	#Read File
	df_gpu=parse_csv_file(args.gpu_file)
	#Produce timeline plot
	event_names, start_times, end_times = get_start_end_times(df_gpu)
	timeline_plot = create(event_names, start_times, end_times, title = 'Timeline Plot', plot_height = 500, plot_width=1200, x_axis_label = 'Time (in ns)', y_axis_label = 'GPU Kernels')
	#Produce Tabular plot
	df_gpu_table = get_table_stats(df_gpu)
	heading, table_plot = create_table(df_gpu_table, heading = 'GPU Calls Summary')
	#Output all GPU calls Plots
	save(column(timeline_plot,heading, table_plot))

	#FIM Log File Visualization
	output_file(filename=cpu_output, title='FIM Visualization', mode='inline')
	#Read File
	df_cpu,df_cpu_buf=parse_fim_log_file(args.fim_file)
	#Produce timeline plot
	event_names, start_times, end_times = get_start_end_times(df_cpu, 'ModuleName', 'BeginTime_us', 'EndTime_us')
	timeline_plot = create(event_names, start_times, end_times, title = 'Timeline Plot', plot_height = 500, plot_width=1200, x_axis_label = 'Time (in us)', y_axis_label = 'FIM Modules')
	#Produce Tabular plot for Module
	df_cpu_module = get_table_stats(df_cpu, 'ModuleName', 'Duration_us', avg_col = 'Average_us', total_col = 'TotalDuration_us')
	heading_m, table_plot_m = create_table(df_cpu_module, heading = 'Module Summary')
	#Produce Tabular plot for APIs
	df_cpu_api = get_table_stats(df_cpu, 'APIName', 'Duration_us', avg_col = 'Average_us', total_col = 'TotalDuration_us')
	heading_a, table_plot_a = create_table(df_cpu_api, heading = 'API Summary')
	#Produce Tabular plot for Buffers
	heading_b, table_plot_b = create_table(df_cpu_buf, heading = 'FIM Buffers Summary')
	#Output all CPU calls Plots
	save(column(timeline_plot, heading_m, table_plot_m, heading_a, table_plot_a, heading_b, table_plot_b))
	
	#MIOpen Log File Visualization
	output_file(filename=mi_output, title='MIOpen Visualization', mode='inline')
	#Read File
	df_mi=parse_miopen_log_file(args.miopen_file)
	#Produce Tabular plot
	heading, table_plot = create_table(df_mi, heading = 'MIOpen API Calls')
	#Output all GPU calls Plots
	save(column(heading, table_plot))

	#Main Page
	output_file(filename=output_name, title='Profiler Visualization', mode='inline')
	gpu_link = gpu_output.split('/')[-1] #Get file name from the complete path
	cpu_link = cpu_output.split('/')[-1]
	mi_link = mi_output.split('/')[-1]
	#Get the main page
	pages = {cpu_link: get_content('fim'), gpu_link: get_content('gpu'), mi_link: get_content('mi')}
	main_plot = create_main(pages)
	#Output Main Page
	show(main_plot)
