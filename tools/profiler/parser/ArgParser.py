import argparse

def arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-g', '--gpu_file', help='GPU calls csv file name with path', default = 'test/gpu_sample.csv')
	parser.add_argument('-f', '--fim_file', help='FIM log file name with path', default = 'test/FIM.INFO')
	parser.add_argument('-m', '--miopen_file', help='MIOpen log file name with path', default = 'test/mi_log_bert.log')
	parser.add_argument('-o', '--output', help='Output File name with path', default = 'Output_Viz.html')

	args = parser.parse_args()
	return args
