import torch
import torch.nn as nn
from torch.autograd import Function
from py_fim_ops import py_fim_eltwise


class FimEltwiseFunction(Function):
    @staticmethod
    def forward(ctx, input1, input2, operation):
        output = py_fim_eltwise(input1, input2, operation)

        return output

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


class FimEltwise(nn.Module):
    """A nn.module wrapper for py_fim_eltwise function.
    """
    def __init__(self, operation=0):
        super(FimEltwise, self).__init__()
        self.operation = operation
        if operation:
            self.op_t = torch.tensor([1], dtype=torch.int32)  #mul
        else:
            self.op_t = torch.tensor([0], dtype=torch.int32)  #add

    def __repr__(self):
        if self.operation:
            return "Fim Eltwise Mul Layer"
        else:
            return "Fim Eltwise Add Layer"

    def forward(self, input1, input2):
        return FimEltwiseFunction.apply(input1, input2, self.op_t)
