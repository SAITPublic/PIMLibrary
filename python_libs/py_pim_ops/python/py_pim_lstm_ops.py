import torch
import os

rocmpath = os.environ['ROCM_PATH']

torch.ops.load_library(os.path.join(rocmpath,"lib","libpy_pim_lstm.so"))

py_pim_lstm = torch.ops.custom_ops.py_pim_lstm

