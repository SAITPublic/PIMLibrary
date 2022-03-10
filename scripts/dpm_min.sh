#!/bin/bash

declare -i gpus=0

if [ $# -gt 0 ]; then
        gpus=$1
fi

counter=$((gpus-1))

for id in $(seq 0 $counter)
do 
	echo "Setting DPM OFF(MIN) setting for $id"
	sudo ./tools/atitool/atitool -ppdpmforce=GFX,0 -i=$id
 	sudo ./tools/atitool/atitool -ppdpmforce=SOC,0 -i=$id
	sudo ./tools/atitool/atitool -ppdpmforce=FCLK,0 -i=$id
	sudo ./tools/atitool/atitool -ppdpmforce=MCLK,0 -i=$id
done

