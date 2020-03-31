# FIMLibrary

FIM Runtime Library and Tools


# How to build (Debug mode)

project_root$ mkdir build

project_root$ cd build

project_root/build$ cmake ..

project_root/build$ make -j


# How to build (Release mode)

project_root$ mkdir build

project_root$ cd build

project_root/build$ cmake -DCMAKE_BUILD_TYPE=Release ..

project_root/build$ make -j


# How to install

- please build as "Release mode" if you want to install libFimRuntime.so

project_root/build$ sudo make install


# How to run FimIntegrationTests

project_root/build$ ./examples/FimIntegrationTests
