set -e
present_working_directory=`pwd`
#install ROCT-Thunk
git clone --branch roc-3.0.x git@github.sec.samsung.net:FIM/ROCT-Thunk-Interface.git
cd ROCT-Thunk-Interface
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm ..
make -j8
sudo make install

cd $present_working_directory

#install ROCR-Runtime
git clone --branch roc-3.0.x git@github.sec.samsung.net:FIM/ROCR-Runtime.git
cd ROCR-Runtime/src
mkdir build
cd build
cmake -DHSAKMT_INC_PATH:STRING="/home/user/fim-workspace/ROCT-Thunk-Interface/include" -DHSAKMT_LIB_PATH:STRING="/opt/rocm/lib" -DCMAKE_INSTALL_PREFIX=/opt/rocm ..
make -j8
sudo make install

cd $present_working_directory

#install hcc
git clone --recursive -b roc-3.0.x git@github.sec.samsung.net:FIM/hcc.git
cd hcc
mkdir -p build; cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/rocm ..
make -j8
sudo make install


cd $present_working_directory


#install rocm-cmake
git clone -b rocm-3.0.0 git@github.sec.samsung.net:FIM/rocm-cmake.git
cd rocm-cmake
mkdir build
cd build
cmake ..
sudo cmake --build . --target install

cd $present_working_directory

#install rocminfo
git clone -b roc-3.0 git@github.sec.samsung.net:FIM/rocminfo.git
cd rocminfo
mkdir build
cd build
cmake -DROCM_DIR=/opt/rocm -DCMAKE_INSTALL_PREFIX=/opt/rocm ..
make -j8
sudo make install

cd $present_working_directory

#install ROCm-Compiler Support
git clone -b roc-3.0.x git@github.sec.samsung.net:FIM/ROCm-CompilerSupport.git
cd ROCm-CompilerSupport/lib/comgr
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/opt/rocm/hcc/lib;/home/user/fim-workspace/hcc/build/llvm-project/llvm/lib/cmake/llvm;/home/user/fim-workspace/hcc/build/lib/cmake/clang;/home/user/fim-workspace/hcc/build;/opt/rocm"  -DCMAKE_INSTALL_PREFIX=/opt/rocm ..
make -j8
sudo make install

cd $present_working_directory

#install hip
git clone -b roc-3.0.x git@github.sec.samsung.net:FIM/HIP.git
cd HIP
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm ..
make -j8
sudo make install

cd $present_working_directory

#install miopen dependency
git clone -b roc-3.0.x git@github.sec.samsung.net:FIM/MIOpen.git
cd MIOpen
sudo cmake -P install_deps.cmake --prefix /opt/rocm

cd $present_working_directory

#install rocblas
git clone -b MIOpen-3.0 git@github.sec.samsung.net:FIM/rocBLAS.git
cd rocBLAS
mkdir -p build/release
cd build/release
CXX=/opt/rocm/bin/hcc cmake -DCMAKE_PREFIX_PATH="/opt/rocm" -DCMAKE_INSTALL_PREFIX=/opt/rocm ../..
make -j8
sudo make install

cd $present_working_directory

#install FIMLibrary
git clone git@github.sec.samsung.net:FIM/FIMLibrary.git
cd FIMLibrary
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm ..
make -j8
sudo make install

cd $present_working_directory

#install MIOpen
cd MIOpen
mkdir build
cd build
CXX=/opt/rocm/bin/hcc cmake -DMIOPEN_BACKEND=HIP -DCMAKE_PREFIX_PATH="/opt/rocm" -DCMAKE_INSTALL_PREFIX=/opt/rocm ..
make -j8
sudo make install
