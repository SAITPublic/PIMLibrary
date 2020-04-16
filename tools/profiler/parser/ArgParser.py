import argparse

def arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_file', help='Input File name with path', default = 'test/fim_add_prof.csv')
	parser.add_argument('-o', '--output_file', help='Output File name with path', default = 'Output_Viz.html')
	
	args = parser.parse_args()
	return args
