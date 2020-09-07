import re
import argparse

file_name = 'uif_pim_traces_v7_09_02/channel_31.uif'
#file_name = 'test'

act_sid = 0
act_bank = 0
act_row = 0
act_pseudo_ch = 0

rw_sid = 0
rw_bank = 0
rw_pseudo_ch = 0
rw_col = 0

parser = argparse.ArgumentParser()
parser.add_argument('--path', required=False)
parser.add_argument('--sid', required=False, type=int)
parser.add_argument('--pc' , required=False, type=int)
args = parser.parse_args()

if (args.path != None) :
	file_name = args.path


def addressMapping(func, byte, physicalAddress: int) :

    #pa = physicalAddress
    #print("pa:", pa, end=" ")
    newTransactionChan = 0
    newTransactionRank = 0
    newTransactionBank = 0
    newTransactionRow = 0
    newTransactionColumn = 0

    tempA = 0
    tempB = 0
    transactionSize = 32
    transactionMask = 31 
    channelBitWidth = 1
    rankBitWidth = 1
    bankBitWidth = 4
    bankgroupBitWidth = 2
    rowBitWidth = 14
    colBitWidth = 7
    byteOffsetWidth = 3;
    colLowBitWidth = 2;
    colHighBitWidth = 5;
    
    col_low_width = 2;
    ba_low_width = 1;
    col_high_width = 3
    ba_high_width = 1
    
    physicalAddress >>= byteOffsetWidth;
    physicalAddress >>= colLowBitWidth;

    tempA = physicalAddress
    physicalAddress = physicalAddress >> 1
    tempB = physicalAddress << 1
    newTransactionColumn = tempA ^ tempB

    tempA = physicalAddress;
    physicalAddress = physicalAddress >> 1
    tempB = physicalAddress << 1
    newTransactionChan = tempA ^ tempB;

    tempA = physicalAddress;
    physicalAddress = physicalAddress >> (col_low_width - 1)
    tempB = physicalAddress << (col_low_width - 1)
    newTransactionColumn |= (tempA ^ tempB) << 1

    # tempA = physicalAddress;
    # physicalAddress = physicalAddress >> (channelBitWidth - 1)
    # tempB = physicalAddress << (channelBitWidth - 1)
    # newTransactionChan |= (tempA ^ tempB) << 1
        
    tempA = physicalAddress
    physicalAddress = physicalAddress >> ba_low_width
    tempB = physicalAddress << ba_low_width
    newTransactionBank = tempA ^ tempB

    tempA = physicalAddress
    physicalAddress = physicalAddress >> bankgroupBitWidth
    tempB = physicalAddress << bankgroupBitWidth
    newTransactionBank |= (tempA ^ tempB) << (bankBitWidth - bankgroupBitWidth)

    tempA = physicalAddress
    physicalAddress = physicalAddress >> ba_high_width
    tempB = physicalAddress << ba_high_width
    newTransactionBank |= (tempA ^ tempB) << ba_low_width

    tempA = physicalAddress
    physicalAddress = physicalAddress >> col_high_width
    tempB = physicalAddress << col_high_width
    newTransactionColumn |= (tempA ^ tempB) << col_low_width

    tempA = physicalAddress
    physicalAddress = physicalAddress >> rowBitWidth
    tempB = physicalAddress << rowBitWidth
    newTransactionRow = tempA ^ tempB

    tempA = physicalAddress
    physicalAddress = physicalAddress >> rankBitWidth
    tempB = physicalAddress << rankBitWidth
    newTransactionRank = tempA ^ tempB


    if ( newTransactionChan==0 and newTransactionRank==0  ) :
        print("{:<9}".format(func), end=" ")
        print(byte, end=" ")
        print("ch: {:<1}".format(newTransactionChan), end=" ")
        print("ra:", newTransactionRank, end=" ")
        print("ba: {:<2}".format(newTransactionBank), end=" ")
        print("row:{:<6}".format(hex(newTransactionRow)), end=" ")
        print("col:", newTransactionColumn)

with open(file_name) as f:
    for line in f:
        inner_list = re.split(':|=| |\n', line)
        inner_list = ' '.join(inner_list).split()
        #for a in inner_list:
        #print(a,end='   ')
        #print(inner_list[1]," ", inner_list[2]," ",  inner_list[6]," : ", int(inner_list[6],0), " : ", bin(int(inner_list[6],0)), end='  ') 
        #print()    
        addressMapping(inner_list[1], inner_list[2], int(inner_list[6],0))

