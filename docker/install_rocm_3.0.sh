set -e
present_working_directory=`pwd`

#install miopen dependency
git clone -b roc-3.0.x git@github.sec.samsung.net:FIM/MIOpen.git
cd MIOpen
sudo cmake -P install_deps.cmake --prefix /opt/rocm

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
