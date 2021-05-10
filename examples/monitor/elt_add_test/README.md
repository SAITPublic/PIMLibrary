# Eltwise-add Test

This test app is implemented based on 32GB memory with 32 channel.
You need to reserve 16GB memory size using ROCK patch before execute this test app.

This is the test code for eltwise-add operation on PIM.
The kernel includes just adding part of the pim-elt-add.
It can be used to verify memory transactions are generated properly.
elt_add_pim.cpp requires a driver patch for PIM to get a base physical address of PIM area.

# Install ROCT and ROCK

- need to install rock2.9 /w memory patch
- need to install roct3.0 /w memory patch

# How to build

$ mkdir build; cd build

$ cmake ..

$ make

# How to run

$ ./elt_add.out
