import torch

torch.ops.load_library("/opt/rocm/lib/libpy_fim_gemv.so")
py_fim_gemv = torch.ops.custom_ops.py_fim_gemv
