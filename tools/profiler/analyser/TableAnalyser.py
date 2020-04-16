import pandas as pd

def get_table_stats(df, dur_col_name = 'DurationNs'):
	'''Function to get Average and Total Time duration of calls
	   df = pandas dataframe
	   dur_col_name = Column name containg duration of calls
	   Return dataframe containg Total and Average time of kernel calls
	'''
	df = df[['KernelName', dur_col_name]]

	#Calculate Average
	avg_df = df.groupby('KernelName').mean()
	avg_df.rename(columns={dur_col_name: 'AverageNs'}, inplace=True)
	#Calculate Total
	total_df = df.groupby('KernelName').sum()
	total_df.rename(columns={dur_col_name: 'TotalDurationNs'}, inplace=True)

	#Merge and reset Index
	df = pd.merge(total_df,avg_df, on = 'KernelName', sort=False)
	df.reset_index(inplace=True)
	df.index = range(len(df))
	return df
