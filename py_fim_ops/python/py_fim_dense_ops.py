import torch

torch.ops.load_library("/opt/rocm/lib/libpy_fim_dense.so")
py_fim_dense = torch.ops.custom_ops.py_fim_dense

