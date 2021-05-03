import torch
import os
path = os.environ['ROCM_PATH']

torch.ops.load_library(os.path.join(path,"lib","libpy_pim_gemv.so"))
py_pim_gemv = torch.ops.custom_ops.py_pim_gemv
