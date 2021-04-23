import torch
import os


path = os.environ['ROCM_PATH']

torch.ops.load_library(os.path.join(path,"lib","libpy_fim_dense.so"))
py_fim_dense = torch.ops.custom_ops.py_fim_dense

