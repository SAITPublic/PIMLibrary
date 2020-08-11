import string
import re


file_name = 'uif_output_traces/umc15device_die0_0/cmd_debug_A_cha.log'

list_of_lists = []

act_sid = 0
act_bank = 0
act_row = 0
act_pseudo_ch = 0

rw_sid = 0
rw_bank = 0
rw_pseudo_ch = 0
rw_col = 0

def convert_param(func, clock, addr, time) :

    global act_sid
    global act_bank
    global act_row
    global act_pseudo_ch

    global rw_sid
    global rw_bank
    global rw_col
    global rw_pseudo_ch

    int_addr = int(addr, 0)

    if ( func == "ACTCMD") :
        if ( clock == "R_r0" ) :
            act_sid = (int_addr >> 2) & 0b1
            act_bank = (int_addr >> 3)
        if ( clock == "R_f0" ) :
            act_bank |= ((int_addr >> 5) & 0b1) << 3
            act_pseudo_ch |= (int_addr >> 3) & 0b1 
            act_row |= (int_addr & 0b11 ) << 11
            act_row |= ((int_addr >> 4) & 0b1) << 13
        if ( clock == "R_r1" ) :
            act_row |= int_addr <<5
        if ( clock == "R_f1" ) :
            act_row |= (int_addr & 0b11) 
            act_row |= ((int_addr >> 3) & 0b111) << 2
            print("%s  SID:%s  PC:%s  BA:%2s  ROW:%6s    @ %s" %(func, act_sid, act_pseudo_ch, act_bank, hex(act_row), time) )
            act_row = act_bank = act_sid = act_pseudo_ch = 0
    else :
        if ( clock == "C_r") :      
            rw_bank |= (int_addr >> 4) 
        if ( clock == "C_f") :
            rw_pseudo_ch |= (int_addr >> 7) & 0b1
            rw_sid |= int_addr & 0b1
            rw_col |= (int_addr >> 1) & 0b1 
            rw_col |= ((int_addr >> 3) & 0b1111) << 1
            print("%6s  SID:%s  PC:%s  BA:%2s  COL:%6s    @ %s" %(func, rw_sid, rw_pseudo_ch, rw_bank, rw_col, time) )
            rw_col = rw_bank = rw_sid = rw_pseudo_ch = 0
    

with open(file_name) as f:
    for line in f:
        inner_list = re.split(':|=| |\n', line)
        inner_list = ' '.join(inner_list).split()

        convert_param(inner_list[3], inner_list[4], inner_list[5], inner_list[8])



