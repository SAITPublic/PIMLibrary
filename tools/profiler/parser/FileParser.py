import pandas as pd

def parse_csv_file(file_name, cols=None):
	'''Parses the csv stat file
		file_name = name of the file
		cols = List of columns to return in dataframe. If cols=None (default), then it return all columns.
		Returns pandas dataframe
	'''
	df=pd.read_csv(file_name)
	if cols:
		df=df[cols]

	return df
