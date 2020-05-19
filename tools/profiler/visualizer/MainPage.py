from bokeh.io import show, output_file
from bokeh.models import Div, Panel, Tabs, ImageURL, Plot, Range1d, ColumnDataSource
from bokeh.layouts import column, row

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
	#Heading
	div_text = '<H1>'+ heading + '</H1>'
	div_heading = Div(text=div_text, sizing_mode="stretch_width" )

	#Logo
	xdr = Range1d()
	ydr = Range1d()
	image_plot = Plot(title=None, x_range=xdr, y_range=ydr, plot_width=600, plot_height=50,  min_border=0, min_border_top=5,min_border_right = 0, outline_line_color=None, toolbar_location=None)
	source = ColumnDataSource(dict(
    	url = ['assets/SAIT_logo.png'],
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

	tabs = Tabs(tabs=tabs, background = '#b3e6ff')

	#CSS template
	template='''
    {% block postamble %}
    <style>
    .bk-root .bk-tabs-header.bk-above .bk-tab {
        border-width: 3px 1px 0px 1px;
        border-radius: 10px 10px 0 0;
    }
    .bk-root .bk-tabs-header .bk-tab.bk-active {
        color: black;
        background-color: white;
        border-color: black;
    }
    .bk-root .bk-tabs-header .bk-tab:hover {
        background-color: #f2f2f2;
    }
    .bk-root .bk-tabs-header .bk-tab {
        padding: 4px 8px;
        border: solid;
        white-space: nowrap;
        cursor: pointer;
        border-color: gray;
    }
    </style>
    {% endblock %}
    '''
	return column(row(image_plot,div_heading),tabs), template

if __name__ == '__main__':

	#Testing the create_main function with random data
	plots = {'title1': Div(text='Test1'), 'title2': Div(text='Test2')}
	output_file(filename='MainViz.html', title='Viz',mode='inline')

	#call the create_main function to get main page
	main = create_main(plots)
	show(main)
