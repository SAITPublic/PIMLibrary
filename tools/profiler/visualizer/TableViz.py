import pandas as pd

def create_table(df, heading = '', output_file  = 'table_plot.html', cols = None):
	'''Outputs an HTML file containing data in tabular format
	df = Pandas dataframe contaning data
	heading = Table heading
	output_file = output Filename
	cols =  List of columns to display in output. If None, whole dataframe is produced as output
	'''

    #render dataframe as html
	if cols:
		html_file = df[cols].to_html()
	else:
		html_file = df.to_html()

	#write html to file
	text_file = open(output_file, 'w')
	text_file.write("<center><H1>"+ heading +"</H1>")
	text_file.write(html_file)
	text_file.write("</center>")
	text_file.close()

if __name__ == '__main__':

	#Testing the create_table function with random data
	data = [['kernel1', 120], ['kernel2', 150], ['kernel3', 140]]

	# Create the pandas DataFrame
	df = pd.DataFrame(data, columns = ['Name', 'Average_Time'])
	create_table(df, 'Test Table', 'testOutput.html',['Name','Average_Time'])
