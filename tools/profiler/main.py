from bokeh.plotting import curdoc
from bokeh.models import Panel, Tabs
import copy

from visualizer.TableViz import create_table, create_table_logs
from visualizer.TimelineViz import create
from visualizer.MainPage import create_main
from visualizer.Config import get_config
from parser.ArgParser import arg_parser
from parser.FileParser import parse_csv_file, parse_pim_log_file, parse_miopen_log_file
from analyser.TimelineAnalyser import get_start_end_times, modify_plot_data
from analyser.TableAnalyser import get_table_stats

# Callback function to change timeline plot according to user input
def slider_callback(attr, old, new):
	start = int(new[0])
	end = int(new[1])
	start_time = df_cpu_all_logs['Time (in ms)'][start]
	end_time = df_cpu_all_logs['Time (in ms)'][end]
	modify_plot_data(glyphs, original_glyphs_data, start_time, end_time)

#Parse the Arguments
args = arg_parser()

#GPU File Visualization
#Read File
df_gpu=parse_csv_file(args.gpu_file)
#Produce timeline plot
event_names, start_times, end_times, tag_names = get_start_end_times(df_gpu)
timeline_plot,_ = create(event_names, start_times, end_times, tag_names, title = 'Timeline Plot', x_axis_label = 'Time (in ms)', y_axis_label = 'GPU Kernels', border_color = get_config('timeline_border_fill_colour'))
#Produce Tabular plot
df_gpu_table = get_table_stats(df_gpu)
table_plot = create_table(df_gpu_table, heading = 'GPU Calls Summary')
#Create tabs for GPU calls Plots
tab_gpu_timeline = Panel(child=timeline_plot, title="GPU Timeline Plot")
tab_gpu_table = Panel(child=table_plot, title="GPU Calls Stats")
tabs_gpu = Tabs(tabs=[tab_gpu_timeline, tab_gpu_table], background = get_config('lower_tab_bg'), width = get_config('lower_tab_width'), height=get_config('lower_tab_height'))

#PIM Log File Visualization
#Read File
df_cpu,df_cpu_buf, df_cpu_all_logs=parse_pim_log_file(args.pim_file)
#Produce timeline plot
event_names, start_times, end_times, tag_names = get_start_end_times(df_cpu, 'Module Name', pim_processing=True)
timeline_plot, glyphs = create(event_names, start_times, end_times, tag_names, title = 'Timeline Plot', x_axis_label = 'Time (in ms)', y_axis_label = 'PIM Modules', border_color = get_config('timeline_border_fill_colour'), pim_plot=True)
#Save the original data of timeline plot
original_glyphs_data = []
for glyph in glyphs:
	glyph_data = copy.deepcopy(dict(glyph.data_source.data))
	original_glyphs_data.append(glyph_data)
#Produce Tabular Plot for all APIs
table_plot_all_logs, range_slider = create_table_logs(df_cpu_all_logs)
range_slider.on_change('value', slider_callback)
#Produce Tabular plot for Module stats
df_cpu_module = get_table_stats(df_cpu, 'Module Name')
table_plot_m = create_table(df_cpu_module, heading = 'Module Summary')
#Produce Tabular plot for APIs stats
df_cpu_api = get_table_stats(df_cpu, 'API Name')
table_plot_a = create_table(df_cpu_api, heading = 'API Summary')
#Produce Tabular plot for Buffers
table_plot_b = create_table(df_cpu_buf, heading = 'PIM Buffers Summary')
#Create tabs for CPU calls Plots
tab_pim_timeline = Panel(child=timeline_plot, title="PIM Timeline Plot")
tab_pim_logs = Panel(child=table_plot_all_logs, title = "PIM Logs")
tab_pim_module = Panel(child=table_plot_m, title="PIM Module Calls Stats")
tab_pim_api = Panel(child=table_plot_a, title="PIM API Calls Stats")
tab_pim_buffer = Panel(child=table_plot_b, title="PIM Buffer Calls")
tabs_pim = Tabs(tabs=[tab_pim_timeline, tab_pim_logs, tab_pim_module, tab_pim_api, tab_pim_buffer], background = get_config('lower_tab_bg'), width = get_config('lower_tab_width'), height=get_config('lower_tab_height'))

#MIOpen Log File Visualization
#Read File
df_mi=parse_miopen_log_file(args.miopen_file)
#Produce Tabular plot
table_plot = create_table(df_mi, heading = 'MIOpen API Calls')
#Create tabs for MIOpen Plot
tab_mi_table = Panel(child=table_plot, title="MIOpen Function Calls")
tabs_mi = Tabs(tabs=[tab_mi_table], background = get_config('lower_tab_bg'), width = get_config('lower_tab_width'), height=get_config('lower_tab_height'))

#Main Page
#Get the main page
tabs = {'GPU Profile Data': tabs_gpu, 'PIM SDK Profile Data': tabs_pim, 'MIOpen APIs Profile Data': tabs_mi}
main_plot = create_main(tabs)
#Output Main Page
curdoc().add_root(main_plot)
curdoc().title = 'Profiler Visualization'
