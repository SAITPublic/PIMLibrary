import pandas as pd
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource, Div, RangeSlider
from bokeh.layouts import column, row

from visualizer.Config import get_config

def create_table(df, heading = 'Summary Table', cols = None, width = get_config('table_plot_width'), height=get_config('table_plot_height')):
	'''Creates a bokeh Tabular plot
	   df = Pandas dataframe contaning data
	   heading = Table heading
	   cols =  List of columns to display in output. If None, whole dataframe is produced as output
	   width = Table width
	   Returns table plot with heading
	'''

	heading_div = Div(text='<H3>'+ heading + '</H3>')
    #Filter column from dataframe
	if cols:
		df = df[cols]

	Columns = [TableColumn(field=Ci, title=Ci) for Ci in df.columns] # bokeh columns
	table_plot = DataTable(columns=Columns, source=ColumnDataSource(df), fit_columns = True, width = width, height=height)# bokeh table

	return column(children=[heading_div, table_plot])

def create_table_logs(df, heading = 'Logs', cols = None, width = get_config('table_plot_width'), height=get_config('table_plot_height')):
	'''Creates a Table for PIM logs to select start and end logs
	   df = Pandas dataframe contaning data
	   heading = Table heading
	   cols =  List of columns to display in output. If None, whole dataframe is produced as output
	   width = Table width
	   Returns table plot with heading
	'''

	heading_div = Div(text='<H3>'+ heading + '</H3>')
    #Filter column from dataframe
	if cols:
		df = df[cols]

	Columns = [TableColumn(field=Ci, title=Ci) for Ci in df.columns] # bokeh columns
	table_plot = DataTable(columns=Columns, source=ColumnDataSource(df), fit_columns = True, width = int(width*0.75), height=height)# bokeh table
	range_slider = RangeSlider(start=0, end=(len(df.index)-1), value=(0,len(df.index)-1), step=1, title="Select range of Logs to display in timeline", width=int(width*0.85), margin = (0,0,0,30))
	return column(row(heading_div, range_slider), table_plot), range_slider


if __name__ == '__main__':

	#Testing the create_table function with random data
	data = [['kernel1', 120], ['kernel2', 150], ['kernel3', 140]]
	# Create the pandas DataFrame
	df = pd.DataFrame(data, columns = ['Name', 'Average_Time'])

	output_file(filename='testTableViz.html', title='TableViz',mode='inline')
	#call the create_table function to get tabular plot
	table_plot = create_table(df, 'Test Table', ['Name','Average_Time'])
	show(table_plot)
