import copy

def get_start_end_times(df, name_col='Kernel Operation', start_col = 'begin', end_col = 'end', pim_processing=False):
	'''Function to extract start and end Times of events from pandas dataframe
		df = Pandas dataframe contaning data
		name_col = Column containg name of events to be plotted (KernelName,ModuleName,APIName, etc)
		start_col = Column contaning start times
		end_col = Column contaning end times
		pim_processing = If the df being called is of PIM calls
		Returns list of event names, start_times and end_times
	'''
	event_start_times = []
	event_end_times = []
	event_names = []
	event_func_names = []

	if(pim_processing):
		round_digit = 3
	else:
		round_digit = 6

	name_groups = df.groupby(name_col)
	for name, group in name_groups:		
		#Add APIs of pim_runtime_api.cpp in case of PIM logs
		if(name.startswith('pim_runtime_api')):
			func_groups = group.groupby('API Name')
			name_list = []
			start_list = []
			end_list = []
			for f_name, f_group in func_groups:
				name_list.append(f_name[20:70]) #Remove initial pim_runtime_api.cpp from name
				f_start_list = f_group[start_col].tolist()
				f_start_list = [round(float(x),round_digit) for x in f_start_list]
				f_end_list = f_group[end_col].tolist()
				f_end_list = [round(float(x),round_digit) for x in f_end_list]
				start_list.append(f_start_list)
				end_list.append(f_end_list)

		else:
			start_list = group[start_col].tolist()
			start_list = [[round(float(x),round_digit) for x in start_list]]
			end_list = group[end_col].tolist()
			end_list = [[round(float(x),round_digit) for x in end_list]]
			name_list = [name]

		event_start_times.append(start_list)
		event_end_times.append(end_list)
		event_func_names.append(name_list)
		event_names.append(name)

	return event_names, event_start_times, event_end_times, event_func_names

def modify_plot_data(glyphs, original_glyphs_data, start_time, end_time):
	for i in range(len(original_glyphs_data)):
		new_data = copy.deepcopy(dict(original_glyphs_data[i]))
		for j in range(len(new_data['left'])):
			if (new_data['right'][j] < start_time) or (new_data['left'][j] > end_time) :
				new_data['right'][j] = start_time
				new_data['left'][j] = start_time
				new_data['names'][j] = ''
			elif new_data['right'][j] > end_time :
				new_data['right'][j] = end_time
			elif new_data['left'][j] < start_time :
				new_data['left'][j] = start_time
		glyphs[i].data_source.data = new_data
