#!/bin/bash

declare -i gpus=0

if [ $# -gt 0 ]; then
        gpus=$1
fi

counter=$((gpus-1))

for id in $(seq 0 $counter)
do 
	echo "Setting DPM MAX (MI100) setting for $id"
	sudo ./tools/atitool/atitool -ppdpmforce=gfx,15 -i=$id
	sudo ./tools/atitool/atitool -ppdpmforce=soc,7 -i=$id
	sudo ./tools/atitool/atitool -ppdpmforce=fclk,7 -i=$id
	sudo ./tools/atitool/atitool -ppdpmforce=mclk,3 -i=$id
done
