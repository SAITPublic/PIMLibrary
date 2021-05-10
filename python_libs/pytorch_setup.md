# PIM-PyTorch

### Setup
__Step1__: Docker Setup
```
./docker_pim.sh pim-rocm4.0:tf2.4-latest-dev <mapping folder>
```

__Step2__: Installing Pytorch
#### rocPRIM
```
git clone https://github.com/ROCmSoftwarePlatform/rocPRIM.git

cd rocPRIM

git checkout tags/rocm-4.0.0

git checkout -b rocm-4.0.0

mkdir build && cd build

CXX=/opt/rocm-4.0.0/bin/hipcc cmake .. 
#Note dependencies are gtest and googlebench which can be skipped(for systems without github access) or built

make -j4

sudo make install

cd ../../
```

#### rocThrust
```
git clone https://github.com/ROCmSoftwarePlatform/rocThrust.git

cd rocThrust

git checkout tags/rocm-4.0.0

git checkout -b rocm-4.0.0

mkdir build && cd build

CXX=/opt/rocm-4.0.0/bin/hipcc cmake .. #Note dependencies are gtest and googlebench which can be skipped(for systems without github access) or built

make -j4

sudo make install

cd ../../
```


#### Download and setup pytorch

***Setup env***
```
export BUILD_CAFFE2=0

export BUILD_CAFFE2_OPS=0

export USE_ROCM=1
```

***Commands:***
```
git clone -b 1.6 https://github.com/ROCmSoftwarePlatform/pytorch

cd pytorch

git submodule update --init --recursive

pip3 install --trusted-host pypi.org --trusted-host files.pythonhosted.org install pyyaml dataclasses

python3 tools/amd_build/build_amd.py #HIPify CUDA kernels

MAX_JOBS=8 python3 setup.py install --user
```


**Download and setup torchvision(optional)**
```
git clone https://github.com/pytorch/vision.git

cd vision

git checkout tags/v0.7.0

git checkout -b v0.7
##Below step because pyyaml was no intalled

python3 setup.py install --user

```

**Check installation**
```

pip3 list | grep torch
```


**Test with **
```

PYTORCH_TEST_WITH_ROCM=1 python3 pytorch/test/run_test.py --verbosed
```

__Step3__: Setup Torch ENV
```
alias python=python3.6
Line39 in vim /home/user/.local/lib/python3.6/site-packages/torch/share/cmake/Torch/TorchConfig.cmake ---->>>set it to OFF
Line 243 in vim /home/user/.local/lib/python3.6/site-packages/torch/include/c10/macros/Macros.h modify __assert_fail_ function name. 
(This is required as the same function signature is also available in hip)
export HIP_PLATFORM=hipcc
export ROCM_PATH=/opt/rocm-4.0.0
export HCC_HOME=${ROCM_PATH}/hcc
export PYTHONPATH=${ROCM_PATH}/lib
export HIP_PATH=${ROCM_PATH}/hip
export Torch_DIR=/home/user/.local/lib/python3.6/site-packages/torch/share/cmake/Torch
```
__Step4__: Setup PIMLibrary
```
git clone git@github.sec.samsung.net:PIM/PIMLibrary.git

cd PIMLibrary

./scripts/build.sh all -o . -t radeon7 -e

./build/examples/PimIntegrationTests
```

__Step5__: Setup MIOpen
```
 git clone git@github.sec.samsung.net:PIM/MIOpen.git

cd MIOpen

git fetch origin rocm-4.0.0:rocm-4.0.0

git checkout rocm-4.0.0

sudo CXX=/opt/rocm-4.0.0/llvm/bin/clang++ cmake -P install_deps.cmake --minimum --prefix /opt/rocm-4.0.0/

mkdir build; cd build

CXX=/opt/rocm-4.0.0/llvm/bin/clang++ cmake -DMIOPEN_BACKEND=HIP -DCMAKE_PREFIX_PATH=/opt/rocm-4.0.0 -DCMAKE_INSTALL_PREFIX=/opt/rocm-4.0.0 ..

make -j8

sudo make install
```

### Testing
  * Run a sample test case by executing `python examples/pytorch_custom_op/pim_eltwise_add_test.py`
  * Run a sample custom model by executing `python examples/pytorch_custom_op/test_sample_model.py`
