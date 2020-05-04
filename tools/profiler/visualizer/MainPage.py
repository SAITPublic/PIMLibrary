from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource, Div

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
	return (name,desc)

def create_main(pages, heading = 'Profiler Plots', width=1200):
	'''Creates main page containing links to all plots
	   pages = Dictionary contaning hyperlink text and path to webpages
	   heading = Page heading
	   Returns an html div containing main page contents
	'''
	div_text = '<H1>'+ heading + '</H1><br>\n<ul>\n'

	for path, (hl_text, description) in pages.items():
		div_text += '<li style=\"font-size:20px\"> <a href = \"' + path + '\"> ' + hl_text + '</a>\n<br>Contains: <ol>\n'
		for desc in description:
			div_text += '<li style=\"font-size:18px\"> '+ desc +'</li>\n'
		div_text+= '</ol></li><br>'

	div_text = div_text + '</ul>'
	div = Div(text=div_text, width=width)
	return div

if __name__ == '__main__':

	#Testing the create_table function with random data
	pages = {'page1.html':('Text1', ['desc1.1', 'desc1.2']), 'page2.html':('Text2', ['desc2.1', 'desc2.2'])}

	output_file(filename='MainViz.html', title='Viz',mode='inline')

	#call the create_main function to get HTML code for main page
	main_div = create_main(pages)
	show(main_div)
