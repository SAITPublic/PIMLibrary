# FIM-PyTorch

### Setup
__Step1__: Docker Setup
```sh
./docker-fim.sh fim-rocm-3.3:pytorch3.6
sudo -i
```

__Step2__: Setup FIM-SDK repo
```sh
git clone https://github.sec.samsung.net/RIB8-SAIT-INDIA/FIMLibrary_PyTorch.git
Refer README.md to install FIM-SDK and its dependencies(Original Setup).
```

__Step3__: Setup Torch ENV
```sh
alias python=python3.6
Line39 in vim /root/.local/lib/python3.6/site-packages/torch/share/cmake/Torch/TorchConfig.cmake ---->>>set it to OFF
export HIP_PLATFORM=hcc
export ROCM_PATH=/opt/rocm
export HCC_HOME=/opt/rocm/hcc
export PYTHONPATH=/opt/rocm/lib
export HIP_PATH=/opt/rocm/hip
export Torch_DIR=/root/.local/lib/python3.6/site-packages/torch/share/cmake/Torch
Refer README.md to build the FIM-SDK
```

### Testing
  * Run a sample test case by executing `python examples/pytorch_custom_op/fim_eltwise_add_test.py`
  * Run a sample custom model by executing `python examples/pytorch_custom_op/test_sample_model.py`
