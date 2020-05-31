#Function to return configuration values such as dimensions and colours
def get_config(attr):
	vals = {
			'timeline_plot_width' : 1400,
			'timeline_plot_height' : 500,
			'timeline_border_fill_colour' : '#99ff99',
			'table_plot_width' : 1470,
			'table_plot_height' : 430,
			'upper_tab_bg' : '#b3e6ff',
			'lower_tab_bg' : '#99ff99',
			'lower_tab_width' : 1510,
			'lower_tab_height' : 540,
			'sait_logo_width' : 600,
			'sait_logo_height' : 50,
			'sait_logo_url' : 'assets/SAIT_logo.png' 
			}
	return vals[attr]
