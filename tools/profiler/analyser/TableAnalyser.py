import pandas as pd

def get_table_stats(df, name_col='Kernel Operation', dur_col = 'duration', avg_col = 'Average Time (in ms)', total_col = 'Total Duration (in ms)'):
	'''Function to get Average and Total Time duration of calls
	   df = pandas dataframe
	   name_col = Column containing names of events (KernelName, ModuleName, APIName, etc)
	   dur_col = Column contaning duration of calls
	   avg_col = Column name for storing Average call duration values
	   total_col = Column name for storing Total call duration values
	   Return dataframe containg Total and Average time of kernel calls
	'''
	df_dur = df[[name_col, dur_col]]
	#Calculate Average
	avg_df = df_dur.groupby(name_col).mean()
	avg_df.rename(columns={dur_col: avg_col}, inplace=True)
	#Calculate Total
	total_df = df_dur.groupby(name_col).sum()
	total_df.rename(columns={dur_col: total_col}, inplace=True)

	#Merge and reset Index
	df_sum = pd.merge(total_df,avg_df, on = name_col, sort=False)
	df_sum.reset_index(inplace=True)
	df_sum.index = range(len(df_sum))
	return df_sum
