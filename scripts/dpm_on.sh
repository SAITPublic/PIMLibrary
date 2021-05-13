#!/bin/bash

sudo ./tools/atitool/atitool -ppdpmstatus=MCLK,on
sudo ./tools/atitool/atitool -ppdpmstatus=FCLK,on
sudo ./tools/atitool/atitool -ppdpmstatus=SOC,on
sudo ./tools/atitool/atitool -ppdpmstatus=GFX,on
sudo ./tools/atitool/atitool -ppdpmstatus=SCLK,on
sudo ./tools/atitool/atitool -ppdpmstatus=PCIE,on
sudo ./tools/atitool/atitool -ppdpmstatus=UVD,on
sudo ./tools/atitool/atitool -ppdpmstatus=VCE,on
sudo ./tools/atitool/atitool -ppdpmstatus=ACP,on
sudo ./tools/atitool/atitool -ppdpmstatus=SAMU,on
sudo ./tools/atitool/atitool -ppdpmstatus=DCEF,on
sudo ./tools/atitool/atitool -ppdpmstatus=PSP,on

