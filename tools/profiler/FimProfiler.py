from bokeh.io import show, output_file
from bokeh.layouts import column

from visualizer.TableViz import create_table
from visualizer.TimelineViz import create
from parser.ArgParser import arg_parser
from parser.FileParser import parse_csv_file
from analyser.TimelineAnalyser import get_start_end_times
from analyser.TableAnalyser import get_table_stats

if __name__=='__main__':

	#Parse the Arguments
	args = arg_parser()
	input_file = args.input_file
	output_file(filename=args.output_file, title='Visualization', mode='inline')

	#Read File
	df=parse_csv_file(input_file)

	#Produce timeline plot
	event_names, start_times, end_times = get_start_end_times(df)
	timeline_plot = create(event_names, start_times, end_times, title = 'Timeline Plot', plot_height = 500, plot_width=1200)

	#Produce Tabular plot
	df = get_table_stats(df)
	heading, table_plot = create_table(df)

	#Output all Plots
	show(column(timeline_plot,heading, table_plot))
