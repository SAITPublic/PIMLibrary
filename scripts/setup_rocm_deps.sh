#!/bin/sh

set -ex

# Set ROCm Version
ROCM_VER=4.0
BRANCH_NAME=rocm-${ROCM_VER}.x

mkdir -p deps
cd deps

# Download and Install ROCm OpenCL Runtime
OPENCL_DIR=`pwd`/ROCm-OpenCL-Runtime
if [ ! -d "${OPENCL_DIR}" ]; then
    git clone --branch ${BRANCH_NAME} https://github.com/RadeonOpenCompute/ROCm-OpenCL-Runtime.git
fi

# Donwload and Install RocCLR
ROCCLR_DIR=`pwd`/ROCclr
if [ ! -d "${ROCCLR_DIR}" ]; then
    git clone --branch ${BRANCH_NAME} https://github.com/ROCm-Developer-Tools/ROCclr.git
fi
cd ${ROCCLR_DIR}
mkdir -p build && cd build
cmake -DOPENCL_DIR=${OPENCL_DIR} -DCMAKE_INSTALL_PREFIX=/opt/rocm/rocclr ..
make -j4
sudo make install
cd ../..

# Download ans Install HIP
HIP_DIR=`pwd`/HIP
if [ ! -d "${HIP_DIR}" ]; then
    git clone --branch ${BRANCH_NAME} git@github.sec.samsung.net:PIM/HIP.git
fi
cd ${HIP_DIR}
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DHIP_COMPILER=clang -DHIP_PLATFORM=rocclr -DCMAKE_PREFIX_PATH="${ROCCLR_DIR}/build;/opt/rocm" -DCMAKE_INSTALL_PREFIX=/opt/rocm -DHSA_PATH=/opt/rocm ..
make -j8
sudo make install
cd ../..    # deps/


# Donwload and Install HSAKMT
KMT_DIR=`pwd`/ROCT-Thunk-Interface
if [ ! -d "${KMT_DIR}" ]; then
    git clone --branch ${BRANCH_NAME} git@github.sec.samsung.net:PIM/ROCT-Thunk-Interface.git
fi
cd ${KMT_DIR}
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm ..
make -j4
sudo make install
cd ../.. # deps/
