import torch

torch.ops.load_library("/opt/rocm/lib/libpy_fim_init.so")
py_fim_init = torch.ops.custom_ops.py_fim_init
