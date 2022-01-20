#!/bin/bash

for id in {0..1}
do 
	echo "Setting DPM OFF(MIN) setting for $id"
	sudo ./tools/atitool/atitool -ppdpmforce=MCLK,0 -i=$id
	sudo ./tools/atitool/atitool -ppdpmstatus=FCLK,off -i=$id
	sudo ./tools/atitool/atitool -ppdpmstatus=SOC,off -i=$id
	sudo ./tools/atitool/atitool -ppdpmstatus=GFX,off -i=$id
	sudo ./tools/atitool/atitool -ppdpmstatus=SCLK,off -i=$id
	sudo ./tools/atitool/atitool -ppdpmstatus=PCIE,off -i=$id
	sudo ./tools/atitool/atitool -ppdpmstatus=UVD,off -i=$id
	sudo ./tools/atitool/atitool -ppdpmstatus=VCE,off -i=$id
	sudo ./tools/atitool/atitool -ppdpmstatus=ACP,off -i=$id
	sudo ./tools/atitool/atitool -ppdpmstatus=SAMU,off -i=$id
	sudo ./tools/atitool/atitool -ppdpmstatus=DCEF,off -i=$id
	sudo ./tools/atitool/atitool -ppdpmstatus=PSP,off -i=$id
done

