import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Function
from py_fim_ops import py_fim_gemv


class FimGemvFunction(Function):
    @staticmethod
    def forward(ctx, inp, weight, reorder):
        output = py_fim_gemv(inp, weight, reorder)

        return output

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


class FimGemv(nn.Module):
    """A nn.module wrapper for py_fim_gemv function.
    """
    def __init__(self, in_features, out_features, reorder=0):
        super(FimGemv, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.HalfTensor(out_features, in_features))

        if reorder:
            self.reorder = torch.tensor([1], dtype=torch.int32)
        else:
            self.reorder = torch.tensor([0], dtype=torch.int32)

    def __repr__(self):
        return "Fim Gemv Layer"

    def forward(self, inp):
        return FimGemvFunction.apply(inp, self.weight, self.reorder)
