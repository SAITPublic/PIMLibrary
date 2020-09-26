import torch

torch.ops.load_library("/opt/rocm/lib/libpy_fim_deinit.so")
py_fim_deinit = torch.ops.custom_ops.py_fim_deinit
