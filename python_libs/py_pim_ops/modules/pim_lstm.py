import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Function
from py_pim_ops import py_pim_lstm


class PimLstmFunction(Function):
    @staticmethod
    def forward(ctx, inp, weight, reorder):
        output = py_pim_lstm(inp, weight, reorder)

        return output

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


class PimLstm(nn.Module):
    """A nn.module wrapper for py_pim_gemv function.
    """
    def __init__(self, in_features, out_features, reorder=0):
        super(PimLstm, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.HalfTensor(out_features, in_features))

        if reorder:
            self.reorder = torch.tensor([1], dtype=torch.int32)
        else:
            self.reorder = torch.tensor([0], dtype=torch.int32)

    def __repr__(self):
        return "Pim Gemv Layer"

    def forward(self, inp):
        return PimLstmFunction.apply(inp, self.weight, self.reorder)
