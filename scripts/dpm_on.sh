#!/bin/bash

for id in {0..1}
do
        sudo ./tools/atitool/atitool -ppdpmstatus=MCLK,on -i=$id
        sudo ./tools/atitool/atitool -ppdpmstatus=FCLK,on -i=$id
        sudo ./tools/atitool/atitool -ppdpmstatus=SOC,on -i=$id
        sudo ./tools/atitool/atitool -ppdpmstatus=GFX,on -i=$id
        sudo ./tools/atitool/atitool -ppdpmstatus=SCLK,on -i=$id
        sudo ./tools/atitool/atitool -ppdpmstatus=PCIE,on -i=$id
        sudo ./tools/atitool/atitool -ppdpmstatus=UVD,on -i=$id
        sudo ./tools/atitool/atitool -ppdpmstatus=VCE,on -i=$id
        sudo ./tools/atitool/atitool -ppdpmstatus=ACP,on -i=$id
        sudo ./tools/atitool/atitool -ppdpmstatus=SAMU,on -i=$id
        sudo ./tools/atitool/atitool -ppdpmstatus=DCEF,on -i=$id
        sudo ./tools/atitool/atitool -ppdpmstatus=PSP,on -i=$id
done

