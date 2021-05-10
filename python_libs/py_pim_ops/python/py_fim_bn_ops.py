import torch
import os

rocmpath = os.environ['ROCM_PATH']
torch.ops.load_library(os.path.join(rocmpath,"lib","libpy_pim_bn.so"))
py_pim_bn = torch.ops.custom_ops.py_pim_bn
