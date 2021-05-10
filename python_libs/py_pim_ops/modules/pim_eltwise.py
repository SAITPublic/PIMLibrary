import torch
import torch.nn as nn
from torch.autograd import Function
from py_pim_ops import py_pim_eltwise


class PimEltwiseFunction(Function):
    @staticmethod
    def forward(ctx, input1, input2, operation):
        output = py_pim_eltwise(input1, input2, operation)

        return output

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


class PimEltwise(nn.Module):
    """A nn.module wrapper for py_pim_eltwise function.
    """
    def __init__(self, operation=0):
        super(PimEltwise, self).__init__()
        self.operation = operation
        if operation:
            self.op_t = torch.tensor([1], dtype=torch.int32)  #mul
        else:
            self.op_t = torch.tensor([0], dtype=torch.int32)  #add

    def __repr__(self):
        if self.operation:
            return "Pim Eltwise Mul Layer"
        else:
            return "Pim Eltwise Add Layer"

    def forward(self, input1, input2):
        return PimEltwiseFunction.apply(input1, input2, self.op_t)
