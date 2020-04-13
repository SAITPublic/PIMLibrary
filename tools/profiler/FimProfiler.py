import pandas as pd
import sys

from visualizer.TableViz import create_table
from parser.FileParser import parse_csv_file

if __name__=='__main__':

	file_name = "test/fim_add_prof.stats.csv" #default file
	if(len(sys.argv)>1):
		file_name = sys.argv[1]

	pd.set_option('display.max_columns', None)
	pd.set_option('display.width', None)
	pd.set_option('display.max_colwidth', 150)

	#Read Stat File
	df_stat=parse_csv_file(file_name, ['Name', 'TotalDurationNs',  'AverageNs'])

	#Produce Tabular Output
	create_table(df_stat, heading = "Summary Table", output_file='statTablePlot.html')
