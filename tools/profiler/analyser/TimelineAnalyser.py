def get_start_end_times(df, name_col='Kernel Operation', start_col = 'begin', end_col = 'end', fim_processing=False):
	'''Function to extract start and end Times of events from pandas dataframe
		df = Pandas dataframe contaning data
		name_col = Column containg name of events to be plotted (KernelName,ModuleName,APIName, etc)
		start_col = Column contaning start times
		end_col = Column contaning end times
		fim_processing = If the df being called is of FIM calls
		Returns list of event names, start_times and end_times
	'''
	event_start_times = []
	event_end_times = []
	event_names = []

	name_groups = df.groupby(name_col)
	for name, group in name_groups:
		event_names.append(name[:50])
		start_list = group[start_col].tolist()
		start_list = [int(x) for x in start_list]
		event_start_times.append(start_list)
		end_list = group[end_col].tolist()
		end_list = [int(x) for x in end_list]
		event_end_times.append(end_list)

	#Add APIs of fim_runtime_api.cpp in case of FIM logs
	if(fim_processing):
		api_groups = df.groupby('API Name')
		for name, group in api_groups:
			if(name.startswith('fim_runtime_api')):
				event_names.append(name[20:70]) #Remove initial fim_runtime_api.cpp from name
				start_list = group[start_col].tolist()
				start_list = [int(x) for x in start_list]
				event_start_times.append(start_list)
				end_list = group[end_col].tolist()
				end_list = [int(x) for x in end_list]
				event_end_times.append(end_list)

	return event_names, event_start_times, event_end_times
