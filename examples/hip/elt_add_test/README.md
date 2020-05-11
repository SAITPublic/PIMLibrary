# Eltwise-add Test

This is the test code for eltwise-add operation on FIM.
The kernel includes just adding part of the fim-elt-add.
It can be used to verify memory transactions are generated properly.
elt_add_fim.cpp requires a driver patch for FIM to get a base physical address of FIM area.

# Install ROCT and ROCK

- need to install rock2.9 /w memory patch
- need to install roct3.0 /w memory patch

# How to build

$ mkdir build; cd build

$ cmake ..

$ make

# How to run

$ ./elt_add.out
