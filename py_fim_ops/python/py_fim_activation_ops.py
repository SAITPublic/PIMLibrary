import torch
import os


path = os.environ['ROCM_PATH']

torch.ops.load_library(os.path.join(path,"lib","libpy_fim_activation.so"))

py_fim_activation = torch.ops.custom_ops.py_fim_activation
