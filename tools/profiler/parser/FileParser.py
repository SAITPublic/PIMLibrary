import pandas as pd
import numpy as np

def parse_csv_file(file_name, cols=['Kernel Operation', 'begin', 'end', 'duration']):
	'''Parses csv file
		file_name = name of the file
		cols = List of columns to return in dataframe. If cols=None, then it return all columns.
		Returns pandas dataframe
	'''
	df=pd.read_csv(file_name)
	df.rename(columns = {'KernelName':'Kernel Operation', 'BeginNs':'begin', 'EndNs':'end', 'DurationNs':'duration'}, inplace=True)

	#Set starting time to 0 ms
	for col in ['end','begin']:
		df[col] = df[col] - df['begin'][0]

	for col in ['begin', 'end','duration']:
		df[col] = (df[col]/(10**6))
	
	if cols:
		df=df[cols]
	return df

def parse_pim_log_file(file_name, cols=None):
	'''Parses pim log file
		file_name = name of the file
		cols = List of columns to return in dataframe. If cols=None (default), then it return all columns.
		Returns pandas dataframes for APIs and buffer
	'''
	df = pd.DataFrame([],columns = ['Module Name', 'API Name','begin','end', 'duration'])
	df_buf = pd.DataFrame([],columns = ['Buffer Id', 'Creation Time (in ms)'])
	df_logs = pd.DataFrame([],columns = ['Module', 'API', 'Time (in ms)'])
	with open(file_name, 'r') as f:
		for line in f:
			if(line[0] == 'I'):
				log  = line.split()
				log_time = log[1].split(':')
				log_time_s = int(log_time[0])*3600+int(log_time[1])*60+float(log_time[2])
				log_time_ms = log_time_s*(10**3)
				module_name = log[3].split(':')[0]
				if(log[4] != '[START]' and log[4] != '[END]'):
					if(len(log) == 9 and log[5] == 'time' and log[7] == ':'): #Buffer Logs
						if(log[6] == '(us)'):
							div = 1000
						elif(log[6] == '(ms)'):
							div = 1
						else:
							print('Problem with buffer log: ', log)
						log_name=log[4]
						log_data = {'Buffer Id':log_name, 'Creation Time (in ms)':(float(log[8])/div)}
						df_buf =df_buf.append(log_data, ignore_index=True)
						#Add log to df_logs
						log_data = {'Module':module_name, 'API':log_name, 'Time (in ms)':log_time_ms}
						df_logs = df_logs.append(log_data, ignore_index=True)
				else:
					log_name = log[5]
					api_name = module_name + ': ' + log[5]
					if(log[4] =='[START]'):
						log_data = {'begin':log_time_ms, 'API Name':api_name, 'Module Name':module_name}
						df =df.append(log_data, ignore_index=True)
						#Add log to df_logs
						log_data = {'Module':module_name, 'API':log_name+' [START]', 'Time (in ms)':log_time_ms}
						df_logs = df_logs.append(log_data, ignore_index=True)
					else:
						row_idx = (df.index[(df['API Name'] == api_name) & (pd.isna(df.end))].tolist()[-1])
						df.at[row_idx, 'end'] = log_time_ms #set end time
						#Add log to df_logs
						log_data = {'Module':module_name, 'API':log_name+' [END]', 'Time (in ms)':log_time_ms}
						df_logs = df_logs.append(log_data, ignore_index=True)

	#Set EndTime = BeginTime wherever Endtime is nan (not in log file)
	df['end'] = np.where(pd.isna(df.end), df['begin'], df['end'])
	#Set starting time to 0 ms
	df['end'] = df['end'] - df['begin'][0]
	df['begin'] = df['begin'] - df['begin'][0]
	df_logs['Time (in ms)'] = df_logs['Time (in ms)'] - df_logs['Time (in ms)'][0]
	#Set duration
	df['duration'] = df['end'] - df['begin']
	#Round off to 3 places
	for col in ['begin', 'end','duration']:
		df[col] = (df[col]).astype('float').round(3)
	df_logs['Time (in ms)'] = (df_logs['Time (in ms)']).astype('float').round(3)

	return df, df_buf, df_logs

def parse_miopen_log_file(file_name, cols=None):
	'''Parses MIOpen log file
		file_name = name of the file
		cols = List of columns to return in dataframe. If cols=None (default), then it return all columns.
		Returns pandas dataframe with MIOpen trace with params
	'''
	df = pd.DataFrame([],columns = ['Function Name', 'Parameters'])
	inside_function = False
	func_param = ''
	with open(file_name, 'r') as f:
		for line in f:
			if(line.startswith('MIOpen(HIP):')):
				line = line[13:] #Remove MIOpen from beginning
				if(inside_function): # Append Params if inside function
					if(line.endswith('}\n')):
						inside_function = False
						log_data = {'Function Name': func_name, 'Parameters': func_param}
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
	df, df_buf = parse_pim_log_file('../test/PIM.INFO')
	print (df)
	print (df_buf)
	df = parse_miopen_log_file('../test/mi_log_bert.log')
	print(df)
	df = parse_csv_file('../test/gpu_sample.csv')
	print(df)
