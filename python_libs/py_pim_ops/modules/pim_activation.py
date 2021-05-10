import torch
import torch.nn as nn
from torch.autograd import Function
from py_pim_ops import py_pim_activation


class PimActivationFunction(Function):
    @staticmethod
    def forward(ctx, inp):
        output = py_pim_activation(inp)

        return output

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


class PimActivation(nn.Module):
    """A nn.module wrapper for py_pim_activation function.
    """
    def __init__(self):
        super(PimActivation, self).__init__()

    def __repr__(self):
        return "Pim Activation Layer"

    def forward(self, inp):
        return PimActivationFunction.apply(inp)
