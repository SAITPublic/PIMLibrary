#!/bin/bash

declare -i gpus=0

if [ $# -gt 0 ]; then
        gpus=$1
fi

counter=$((gpus-1))

for id in $(seq 0 $counter)
do 
	echo "Setting DPM MAX (MI50) setting for $id"
	sudo ./tools/atitool/atitool -ppdpmforce=gfx,8 -i=$id
	sudo ./tools/atitool/atitool -ppdpmforce=soc,7 -i=$id
	sudo ./tools/atitool/atitool -ppdpmforce=fclk,7 -i=$id
	sudo ./tools/atitool/atitool -ppdpmforce=mclk,2 -i=$id
done
