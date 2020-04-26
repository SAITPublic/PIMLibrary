import pandas as pd

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
	first_line=True

	with open(file_name, 'r') as f:
		for line in f:
			if(line[0] == 'I'):
				log  = line.split()
				begin_time = log[1].split(':')
				begin_time_s = int(begin_time[0])*3600+int(begin_time[1])*60+float(begin_time[2])
				begin_time_us = begin_time_s*(10**6)
				log_data = {'BeginTime_us':begin_time_us, 'APIName':log[3][:-1], 'ModuleName':log[3].split(':')[0]}
				if(first_line):
					first_line=False
				else:
					prev_log_data['EndTime_us'] = log_data['BeginTime_us']
					prev_log_data['Duration_us'] = prev_log_data['EndTime_us'] - prev_log_data['BeginTime_us']
					df =df.append(prev_log_data, ignore_index=True)
				prev_log_data = log_data
		prev_log_data['EndTime_us'] = log_data['BeginTime_us']
		prev_log_data['Duration_us'] = prev_log_data['EndTime_us'] - prev_log_data['BeginTime_us']
		df =df.append(prev_log_data, ignore_index=True)

	return df

if(__name__=='__main__'):
	#Test the parser with default file
	df = parse_log_file('../test/fim.log')
	print (df)
