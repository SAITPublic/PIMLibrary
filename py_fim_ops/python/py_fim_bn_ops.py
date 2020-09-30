import torch

torch.ops.load_library("/opt/rocm/lib/libpy_fim_bn.so")
py_fim_bn = torch.ops.custom_ops.py_fim_bn
