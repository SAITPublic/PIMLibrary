import argparse

def arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-g', '--gpu_file', help='GPU calls csv file name with path', default = 'test/fim_add_prof.csv')
	parser.add_argument('-c', '--cpu_file', help='CPU calls log file name with path', default = 'test/fim.log')
	parser.add_argument('-a', '--gpu_output', help='GPU Output File name with path', default = 'Output_Viz_GPU.html')
	parser.add_argument('-b', '--cpu_output', help='CPU Output File name with path', default = 'Output_Viz_CPU.html')

	args = parser.parse_args()
	return args
