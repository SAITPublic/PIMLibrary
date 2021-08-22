#!/bin/bash
echo ""
echo ""
echo "             < MIOpen GEMV Benchmark Test >"
echo ""
echo "Workload : GEMV (1x1024) * (1024x4096)"
echo ""
echo "[GPU+HBM mode test]"
echo ""
export ENABLE_PIM=0
export ENABLE_NEXT_PIM=0
MIOpenDriver gemmfp16 -F 1 -m 1 -k 1024 -n 4096 -v 1 -V 0 -i 100
echo ""
echo ""
echo "[GPU+PIM mode test]"
echo ""
export ENABLE_PIM=1
export ENABLE_NEXT_PIM=0
MIOpenDriver gemmfp16 -F 1 -m 1 -k 1024 -n 4096 -v 1 -V 0 -i 100
echo ""
echo ""
echo "[Next GPU+PIM mode test]"
echo ""
export ENABLE_PIM=1
export ENABLE_NEXT_PIM=1
MIOpenDriver gemmfp16 -F 1 -m 1 -k 1024 -n 4096 -v 1 -V 0 -i 100
echo ""
echo ""
