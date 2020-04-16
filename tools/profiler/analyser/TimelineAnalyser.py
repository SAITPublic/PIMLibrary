def get_start_end_times(df, start_col_name = 'BeginNs', end_col_name = 'EndNs'):
	'''Function to extract start and end Times of events from pandas dataframe
		df = Pandas dataframe contaning data
		start_col_name = Name of column contaning start times
		end_col_name = Name of column contaning end times
		Returns list of event names, start_times and end_times
	'''
	event_start_times = []
	event_end_times = []
	event_names = []

	name_groups = df.groupby('KernelName')

	for name, group in name_groups:
		event_names.append(name[0:20])
		start_list = group[start_col_name].tolist()
		start_list = [int(x) for x in start_list]
		event_start_times.append(start_list)
		end_list = group[end_col_name].tolist()
		end_list = [int(x) for x in end_list]
		event_end_times.append(end_list)

	return event_names, event_start_times, event_end_times
