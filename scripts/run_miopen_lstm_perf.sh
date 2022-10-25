#!/bin/bash
echo "             < MIOpen LSTM Benchmark Test >"
echo ""
echo "Workload : LSTM layers for DeepSpeech2 Inference"
echo "           2 seconds voice recognition"
echo "Input (2592), Hidden (1024), 5 layers, 50 steps, bi-directional"
echo ""
echo "[GPU+HBM mode test]"
echo ""
export ENABLE_PIM=0
export ENABLE_NEXT_PIM=0
MIOpenDriver rnnfp16 -F 1 -H 1024 -V 1 -W 2592 -c 1 -k 50 -l 5 -m lstm -n 1 -r 1 -w 1 -i 2

echo ""
echo "[GPU+PIM mode test]"
echo ""
export ENABLE_PIM=1
export ENABLE_NEXT_PIM=0
MIOpenDriver rnnfp16 -F 1 -H 1024 -V 1 -W 2592 -c 1 -k 50 -l 5 -m lstm -n 1 -r 1 -w 1 -i 2
echo ""
echo "[Next GPU+PIM mode test]"
echo ""
export ENABLE_PIM=1
export ENABLE_NEXT_PIM=1
MIOpenDriver rnnfp16 -F 1 -H 1024 -V 1 -W 2592 -c 1 -k 50 -l 5 -m lstm -n 1 -r 1 -w 1 -i 2
