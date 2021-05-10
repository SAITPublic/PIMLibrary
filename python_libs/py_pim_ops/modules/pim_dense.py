import torch
import torch.nn as nn
from torch.autograd import Function
from py_pim_ops import py_pim_dense


class PimDenseFunction(Function):
    @staticmethod
    def forward(ctx, inputs, weights, bias_flag, bias):
        output = py_pim_dense(inputs, weights, bias_flag, bias)

        return output

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


class PimDense(nn.Module):
    """A nn.module wrapper for py_pim_dense function.
    """
    def __init__(self):
        super(PimDense, self).__init__()

    def __repr__(self):
        return "PIM dense layer"

    def forward(self, inputs, weights, bias_flag, bias):
        return PimDenseFunction.apply(inputs, weights, bias_flag, bias)
