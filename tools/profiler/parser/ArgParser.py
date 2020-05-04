import argparse

def arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-g', '--gpu_file', help='GPU calls csv file name with path', default = 'test/fim_add_prof.csv')
	parser.add_argument('-c', '--cpu_file', help='CPU calls log file name with path', default = 'test/FIM.INFO')
	parser.add_argument('-o', '--output', help='Output File name with path', default = 'Output_Viz.html')

	args = parser.parse_args()
	return args
