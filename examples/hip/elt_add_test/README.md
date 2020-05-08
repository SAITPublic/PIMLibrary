# Eltwise-add Test

This is the test code for eltwise-add operation on FIM.
The kernel includes just adding part of the fim-elt-add.
It can be used to verify memory transactions are generated properly.
elt_add_fim.cpp requires a driver patch for FIM to get a base physical address of FIM area.

# How to build

$ make

# How to run

$ ./elt_add.out
