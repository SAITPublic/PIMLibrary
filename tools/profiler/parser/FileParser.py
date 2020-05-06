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

def parse_fim_log_file(file_name, cols=None):
	'''Parses fim log file
		file_name = name of the file
		cols = List of columns to return in dataframe. If cols=None (default), then it return all columns.
		Returns pandas dataframes for APIs and buffer
	'''
	df = pd.DataFrame([],columns = ['ModuleName', 'APIName','BeginTime_us','EndTime_us', 'Duration_us'])
	df_buf = pd.DataFrame([],columns = ['BufferID', 'BufferCreationTime_us'])
	with open(file_name, 'r') as f:
		for line in f:
			if(line[0] == 'I'):
				log  = line.split()
				if(log[4] != '[START]' and log[4] != '[END]'):
					log_data = {'BufferID':log[4], 'BufferCreationTime_us':log[8]}
					df_buf =df_buf.append(log_data, ignore_index=True)
				else:
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

	return df, df_buf

def parse_miopen_log_file(file_name, cols=None):
	'''Parses MIOpen log file
		file_name = name of the file
		cols = List of columns to return in dataframe. If cols=None (default), then it return all columns.
		Returns pandas dataframe with MIOpen trace with params
	'''
	df = pd.DataFrame([],columns = ['FunctionName', 'Parameters'])
	inside_function = False
	func_param = ''
	with open(file_name, 'r') as f:
		for line in f:
			if(line.startswith('MIOpen(HIP):')):
				line = line[13:] #Remove MIOpen from beginning
				if(inside_function): # Append Params if inside function
					if(line.endswith('}\n')):
						inside_function = False
						log_data = {'FunctionName': func_name, 'Parameters': func_param}
						df = df.append(log_data, ignore_index=True)
					else:
						func_param += (line.rstrip('\n').lstrip('\t') + ' ')
				elif(line.endswith('{\n')): #Get function Name from log
					func_name = line[:-2]
					inside_function = True
					func_param = ''

	return df

if(__name__=='__main__'):
	#Test the parser with default file
	df, df_buf = parse_fim_log_file('../test/FIM.INFO')
	print (df)
	print (df_buf)
	df = parse_miopen_log_file('../test/mi_log_bert.log')
	print(df)
