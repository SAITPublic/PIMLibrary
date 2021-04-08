#!/bin/bash

while :
do
  rocm-smi --showmemuse --showuse --showpower --csv >> $1
  sleep 0.01
done
