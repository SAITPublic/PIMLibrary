import sys
from bokeh.io import show, output_file
from bokeh.layouts import column

from visualizer.TableViz import create_table
from parser.FileParser import parse_csv_file

if __name__=='__main__':

	file_name = "test/fim_add_prof.stats.csv" #default file
	if(len(sys.argv)>1):
		file_name = sys.argv[1]

	output_file(filename='Output_Viz.html', title='Table Visualization', mode='inline')

	#Read Stat File
	df_stat=parse_csv_file(file_name, ['Name', 'TotalDurationNs',  'AverageNs'])

	#Produce Tabular Output
	heading, table_plot = create_table(df_stat)
	show(column(heading, table_plot))
