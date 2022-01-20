#!/bin/bash

for id in {0..1}
do 
	echo "Setting DPM OFF(MIN) setting for $id"
	sudo ./tools/atitool/atitool -ppdpmforce=GFX,0 -i=$id
 	sudo ./tools/atitool/atitool -ppdpmforce=SOC,0 -i=$id
	sudo ./tools/atitool/atitool -ppdpmforce=FCLK,0 -i=$id
	sudo ./tools/atitool/atitool -ppdpmforce=MCLK,0 -i=$id
done

