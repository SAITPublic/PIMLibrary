import enum
import math

class ValueType(enum.Enum):
    NORMAL = 0
    NS = 1
    TCK = 2

tCK=0.486
BL=16

class Value:
    def __init__(self, name, value0, value_type0, value1=0, value_type1=ValueType.NORMAL):
        if (value1 != 0):
            self.name = name
            if (value_type0 == ValueType.NS):
                self.value = math.ceil(value0 / tCK) 
            if (value_type1 == ValueType.NS):
                self.value = max (self.value, math.ceil(value1 / tCK)) 
            else:
                self.value = max (self.value, value1) 
                
            self.value_type = ValueType.TCK
        else:
            self.name = name
            self.value = value0
            self.value_type = value_type0
            

    def __str__(self):
        val = self.value
        if (self.value_type == ValueType.NS):
            val = math.ceil(val / tCK)
        if (self.value_type == ValueType.TCK):
        	val = math.ceil(val)
        return (self.name+"="+str(val))

param_list = []
param_list.append( Value("NUM_BANK_GROUPS",1, ValueType.NORMAL))
param_list.append( Value("NUM_BANKS",8, ValueType.NORMAL))
param_list.append( Value("NUM_COLS",1024, ValueType.NORMAL))
param_list.append( Value("NUM_ROWS",65536, ValueType.NORMAL))
param_list.append( Value("NUM_FIM_BLOCKS",8, ValueType.NORMAL))
param_list.append( Value("DEVICE_WIDTH",16, ValueType.NORMAL))
param_list.append( Value("BL",16, ValueType.NORMAL))
if (tCK==0.625):
    param_list.append( Value("RL",28, ValueType.TCK))
    param_list.append( Value("WL",14, ValueType.TCK))
elif (tCK==0.468):
    param_list.append( Value("RL",36, ValueType.TCK))
    param_list.append( Value("WL",18, ValueType.TCK))
    
param_list.append( Value("tCCDS",BL/2, ValueType.TCK))
param_list.append( Value("tCCDL",BL/2, ValueType.TCK))

param_list.append( Value("tRCDRD",18, ValueType.NS, 4, ValueType.TCK))
param_list.append( Value("tRCDWR",18, ValueType.NS, 4, ValueType.TCK))

param_list.append( Value("tRAS",42, ValueType.NS, 3, ValueType.TCK))

if (tCK==0.625):
    param_list.append( Value("tRRDS", 7.5, ValueType.NS, 4, ValueType.TCK))
    param_list.append( Value("tRRDL", 7.5, ValueType.NS, 4, ValueType.TCK))
elif (tCK==0.468):
    param_list.append( Value("tRRDS", 10, ValueType.NS, 4, ValueType.TCK))
    param_list.append( Value("tRRDL", 10, ValueType.NS, 4, ValueType.TCK))

# RAS + RP
param_list.append( Value("tRC",42+18, ValueType.NS, 3+4, ValueType.TCK))
param_list.append( Value("tRP",18, ValueType.NS, 4, ValueType.TCK))

param_list.append( Value("tRTPS",7.5, ValueType.NS, 8, ValueType.TCK))
param_list.append( Value("tRTPL",7.5, ValueType.NS, 8, ValueType.TCK))

param_list.append( Value("tWR",18, ValueType.NS, 6, ValueType.TCK))

param_list.append( Value("tWTRS",10, ValueType.NS, 8, ValueType.TCK))
param_list.append( Value("tWTRL",10, ValueType.NS, 8, ValueType.TCK))

param_list.append( Value("XAW",4, ValueType.NORMAL))

if (tCK==0.625):
    param_list.append( Value("tXAW",40, ValueType.NS))
elif (tCK==0.468):
    param_list.append( Value("tXAW",30, ValueType.NS))

param_list.append( Value("tRTRS",1, ValueType.TCK)) #unknown

param_list.append( Value("tREFI",3904, ValueType.NORMAL))
param_list.append( Value("tREFISB",488, ValueType.NORMAL))
param_list.append( Value("tRFC",280, ValueType.NS))
param_list.append( Value("tRFCSB",140, ValueType.NS))
param_list.append( Value("tXP",7.5, ValueType.NS, 5, ValueType.TCK))
param_list.append( Value("tCKE",7.5, ValueType.NS, 3, ValueType.TCK))
param_list.append( Value("tCMD",1, ValueType.NORMAL))
param_list.append( Value("AL",0, ValueType.NORMAL))

param_list.append( Value("tCK",tCK, ValueType.NORMAL))

param_list.append( Value("IDD0",0, ValueType.NORMAL))
param_list.append( Value("IDD0C",0, ValueType.NORMAL))
param_list.append( Value("IDD0Q",0, ValueType.NORMAL))
param_list.append( Value("IDD1",0, ValueType.NORMAL))
param_list.append( Value("IDD2P",0, ValueType.NORMAL))
param_list.append( Value("IDD2Q",0, ValueType.NORMAL))
param_list.append( Value("IDD2N",0, ValueType.NORMAL))
param_list.append( Value("IDD3Pf",0, ValueType.NORMAL))
param_list.append( Value("IDD3Ps",0, ValueType.NORMAL))
param_list.append( Value("IDD3N",0, ValueType.NORMAL))
param_list.append( Value("IDD3NC",0, ValueType.NORMAL))
param_list.append( Value("IDD3NQ",0, ValueType.NORMAL))
param_list.append( Value("IDD4W",0, ValueType.NORMAL))
param_list.append( Value("IDD4WC",0, ValueType.NORMAL))
param_list.append( Value("IDD4WQ",0, ValueType.NORMAL))
param_list.append( Value("IDD4R",0, ValueType.NORMAL))
param_list.append( Value("IDD4RC",0, ValueType.NORMAL))
param_list.append( Value("IDD4RQ",0, ValueType.NORMAL))

param_list.append( Value("IDD5",0, ValueType.NORMAL))
param_list.append( Value("IDD6",0, ValueType.NORMAL))
param_list.append( Value("IDD6L",0, ValueType.NORMAL))
param_list.append( Value("IDD7",0, ValueType.NORMAL))

param_list.append( Value("Vdd",1, ValueType.NORMAL))
param_list.append( Value("Vddc",1, ValueType.NORMAL))
param_list.append( Value("Vddq",1, ValueType.NORMAL))
param_list.append( Value("Vpp",1, ValueType.NORMAL))

param_list.append( Value("Ealu",0, ValueType.NORMAL))
param_list.append( Value("Ereg",0, ValueType.NORMAL))

for param in param_list:
    print (param)
