import torch

torch.ops.load_library("/opt/rocm/lib/libpy_fim_activation.so")
py_fim_activation = torch.ops.custom_ops.py_fim_activation
