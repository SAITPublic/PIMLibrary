import torch
import torch.nn as nn
from torch.autograd import Function
from py_fim_ops import py_fim_dense


class FimDenseFunction(Function):
    @staticmethod
    def forward(ctx, inputs, weights, bias_flag, bias):
        output = py_fim_dense(inputs, weights, bias_flag, bias)

        return output

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


class FimDense(nn.Module):
    """A nn.module wrapper for py_fim_dense function.
    """
    def __init__(self):
        super(FimDense, self).__init__()

    def __repr__(self):
        return "FIM dense layer"

    def forward(self, inputs, weights, bias_flag, bias):
        return FimDenseFunction.apply(inputs, weights, bias_flag, bias)
