# How to build

$ sudo apt-get install miopen-hip
$ mkdir build
$ cd build
$ CXX=/opt/rocm/hip/bin/hipcc cmake ..
$ make -j8
