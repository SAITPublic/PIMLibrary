#!/bin/bash


for id in {0..1}
do 
	echo "Setting DPM MAX (MI50) setting for $id"
	sudo ./tools/atitool/atitool -ppdpmforce=gfx,8 -i=$id
	sudo ./tools/atitool/atitool -ppdpmforce=soc,7 -i=$id
	sudo ./tools/atitool/atitool -ppdpmforce=fclk,7 -i=$id
	sudo ./tools/atitool/atitool -ppdpmforce=mclk,2 -i=$id
done
