import os
import signal
import sys
import pandas as pd
import numpy as np
import argparse
import fileinput

parser = argparse.ArgumentParser(description='rocm-smi profiling')

parser.add_argument('-i','--input', default="input.txt", help="Input file name with path")
parser.add_argument('-o','--output', default="output", help="Output filename with path")

args = parser.parse_args()

def generate_excel(input_file, output_file):
    # Reading the csv file
    df_new = pd.read_csv(input_file)
      
    # saving xlsx file
    time_cnt = 0

    for i in df_new.index:
        df_new.at[i, "Time (ms)"] = time_cnt
        time_cnt += 10

    GFG = pd.ExcelWriter(output_file + '.xlsx')
    df_new.to_excel(GFG, index = False)
        
    GFG.save()

def clean_generated_file(input_file):
    fin = open(input_file, "rt")
    data = fin.read()
    data = data.replace('device,Average Graphics Package Power (W),GPU use (%),GPU memory use (%)', '')
    fin.close()

    fin = open(input_file, "wt")
    fin.write("Time (ms),Power (W),GPU use (%),MEM use (%)\n" + data)
    fin.close()

    for line in fileinput.FileInput(input_file,inplace=1):
        if line.rstrip():
            print(line)

if __name__ == '__main__':
    print('User arguments {}'.format(args))
    clean_generated_file(args.input)
    generate_excel(args.input, args.output)
