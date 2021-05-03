import torch
import os
path = os.environ['ROCM_PATH']

torch.ops.load_library(os.path.join(path,"lib","libpy_pim_init.so"))

py_pim_init = torch.ops.custom_ops.py_pim_init
