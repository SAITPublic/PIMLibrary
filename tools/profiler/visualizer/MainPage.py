from bokeh.io import show, output_file
from bokeh.models import Div, Panel, Tabs
from bokeh.layouts import column

def get_content(plot_type):
	'''Utility function to get  name and description for different profiler plot pages
	   plot_type = can either be fim/gpu
	   Returns name and description
	'''
	if plot_type == 'fim' :
		name = 'FIM SDK Profile Data'
		desc = ['Timeline plot for FIM SDK executions sequence', 'Stats of FIM Modules execution', 'API Trace and Performance', 'Buffers created']
	elif plot_type == 'gpu' :
		name = 'GPU Profile Data'
		desc = ['Timeline plot of kernel operations', 'Stats of kernel execution']
	elif plot_type == 'mi' :
		name = 'MIOpen APIs Profile Data'
		desc = ['MIOpen API trace and parameters']
	return (name,desc)

def create_main(plots, heading = 'Profiler Plots', width=1200):
	'''Creates main page containing all plots inside tabs
	   plots = Dictionary contaning title and the plots
	   heading = Page heading
	   Returns main page contents
	'''
	div_text = '<H1>'+ heading + '</H1>'
	div_heading = Div(text=div_text, width=width)

	tabs = []
	for title, plot in plots.items():
		div = Div(text='<H2>'+ title + '</H2>', align='center')
		tab = Panel(child=column(div, plot), title=title)
		tabs.append(tab)

	tabs = Tabs(tabs=tabs)
	return column(div_heading,tabs)

if __name__ == '__main__':

	#Testing the create_main function with random data
	plots = {'title1': Div(text='Test1'), 'title2': Div(text='Test2')}
	output_file(filename='MainViz.html', title='Viz',mode='inline')

	#call the create_main function to get main page
	main = create_main(plots)
	show(main)
