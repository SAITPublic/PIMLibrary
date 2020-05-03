import pandas as pd
import numpy as np
def parse_csv_file(file_name, cols=None):
	'''Parses csv file
		file_name = name of the file
		cols = List of columns to return in dataframe. If cols=None (default), then it return all columns.
		Returns pandas dataframe
	'''
	df=pd.read_csv(file_name)
	if cols:
		df=df[cols]

	return df

def parse_log_file(file_name, cols=None):
	'''Parses log file
		file_name = name of the file
		cols = List of columns to return in dataframe. If cols=None (default), then it return all columns.
		Returns pandas dataframe
	'''
	df = pd.DataFrame([],columns = ['ModuleName', 'APIName','BeginTime_us','EndTime_us', 'Duration_us'])
	with open(file_name, 'r') as f:
		for line in f:
			if(line[0] == 'I'):
				log  = line.split()
				if(log[4] != '[START]' and log[4] != '[END]'):
					continue

				log_time = log[1].split(':')
				log_time_s = int(log_time[0])*3600+int(log_time[1])*60+float(log_time[2])
				log_time_us = log_time_s*(10**6)
				module_name = log[3].split(':')[0]
				api_name = module_name + ':' + log[5]

				if(log[4] =='[START]'):
					log_data = {'BeginTime_us':log_time_us, 'APIName':api_name, 'ModuleName':module_name}
					df =df.append(log_data, ignore_index=True)
				else:
					row_idx = (df.index[(df.APIName == api_name) & (pd.isna(df.EndTime_us))].tolist()[-1])
					df.at[row_idx, 'EndTime_us'] = log_time_us #set end time

	#Set EndTime = BeginTime wherever Endtime is nan (not in log file)
	df['EndTime_us'] = np.where(pd.isna(df.EndTime_us), df['BeginTime_us'], df['EndTime_us'])
	df['Duration_us'] = df['EndTime_us'] - df['BeginTime_us'] #set duration

	return df

if(__name__=='__main__'):
	#Test the parser with default file
	df = parse_log_file('../test/FIM.INFO')
	print (df)
	print (df.info())
