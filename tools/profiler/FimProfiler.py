from bokeh.io import show, output_file
from bokeh.layouts import column

from visualizer.TableViz import create_table
from parser.ArgParser import arg_parser
from parser.FileParser import parse_csv_file

if __name__=='__main__':

	#Parse the Arguments
	args = arg_parser()
	input_file = args.input_file

	output_file(filename=args.output_file, title='Table Visualization', mode='inline')

	#Read Stat File
	df_stat=parse_csv_file(input_file, ['Name', 'TotalDurationNs',  'AverageNs'])

	#Produce Tabular Output
	heading, table_plot = create_table(df_stat)
	show(column(heading, table_plot))
