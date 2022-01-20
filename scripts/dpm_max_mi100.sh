#!/bin/bash


for id in {0..1}
do 
	echo "Setting DPM MAX (MI100) setting for $id"
	sudo ./tools/atitool/atitool -ppdpmforce=gfx,15 -i=$id
	sudo ./tools/atitool/atitool -ppdpmforce=soc,7 -i=$id
	sudo ./tools/atitool/atitool -ppdpmforce=fclk,7 -i=$id
	sudo ./tools/atitool/atitool -ppdpmforce=mclk,3 -i=$id
done
