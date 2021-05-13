#!/bin/bash

sudo ./tools/atitool/atitool -ppdpmstatus=MCLK,off
sudo ./tools/atitool/atitool -ppdpmstatus=FCLK,off
sudo ./tools/atitool/atitool -ppdpmstatus=SOC,off
sudo ./tools/atitool/atitool -ppdpmstatus=GFX,off
sudo ./tools/atitool/atitool -ppdpmstatus=SCLK,off
sudo ./tools/atitool/atitool -ppdpmstatus=PCIE,off
sudo ./tools/atitool/atitool -ppdpmstatus=UVD,off
sudo ./tools/atitool/atitool -ppdpmstatus=VCE,off
sudo ./tools/atitool/atitool -ppdpmstatus=ACP,off
sudo ./tools/atitool/atitool -ppdpmstatus=SAMU,off
sudo ./tools/atitool/atitool -ppdpmstatus=DCEF,off
sudo ./tools/atitool/atitool -ppdpmstatus=PSP,off

