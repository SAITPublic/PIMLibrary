from bokeh.io import show, output_file
from bokeh.models import Div, Panel, Tabs, ImageURL, Plot, Range1d, ColumnDataSource
from bokeh.layouts import column, row

from visualizer.Config import get_config

def get_content(plot_type):
	'''Utility function to get  name and description for different profiler plot pages
	   plot_type = can either be pim/gpu
	   Returns name and description
	'''
	if plot_type == 'pim' :
		name = 'PIM SDK Profile Data'
		desc = ['Timeline plot for PIM SDK executions sequence', 'Stats of PIM Modules execution', 'API Trace and Performance', 'Buffers created']
	elif plot_type == 'gpu' :
		name = 'GPU Profile Data'
		desc = ['Timeline plot of kernel operations', 'Stats of kernel execution']
	elif plot_type == 'mi' :
		name = 'MIOpen APIs Profile Data'
		desc = ['MIOpen API trace and parameters']
	return (name,desc)

def create_main(plots, heading = 'Profiler Plots'):
	'''Creates main page containing all plots inside tabs
	   plots = Dictionary contaning title and the plots
	   heading = Page heading
	   Returns main page contents
	'''
	#Heading
	div_text = '<H1>'+ heading + '</H1>'
	div_heading = Div(text=div_text, sizing_mode="stretch_width" )

	#Logo
	xdr = Range1d()
	ydr = Range1d()
	image_plot = Plot(title=None, x_range=xdr, y_range=ydr, plot_width=get_config('sait_logo_width'), plot_height=get_config('sait_logo_height'),  min_border=0, min_border_top=5,min_border_right = 0, outline_line_color=None, toolbar_location=None)
	source = ColumnDataSource(dict(
    	url = [get_config('sait_logo_url')],
		x1  = [0],
		y1  = [0],
	))
	image1 = ImageURL(url="url", x="x1", y="y1", anchor="bottom_left")
	image_plot.add_glyph(source, image1)

	#Tabs
	tabs = []
	for title, plot in plots.items():
		div = Div(text='<H2>'+ title + '</H2>', align='center')
		tab = Panel(child=column(div, plot), title=title)
		tabs.append(tab)

	tabs = Tabs(tabs=tabs, background = get_config('upper_tab_bg'))

	return column(row(image_plot,div_heading),tabs)

if __name__ == '__main__':

	#Testing the create_main function with random data
	plots = {'title1': Div(text='Test1'), 'title2': Div(text='Test2')}
	output_file(filename='MainViz.html', title='Viz',mode='inline')

	#call the create_main function to get main page
	main = create_main(plots)
	show(main)
