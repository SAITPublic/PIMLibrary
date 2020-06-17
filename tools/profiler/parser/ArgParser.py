import argparse

def arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-g', '--gpu_file', help='GPU calls csv file name with path', default = 'test/gpu_sample.csv')
	parser.add_argument('-f', '--fim_file', help='FIM log file name with path', default = 'test/FIM.INFO')
	parser.add_argument('-m', '--miopen_file', help='MIOpen log file name with path', default = 'test/mi_log_bert.log')

	args = parser.parse_args()
	return args
