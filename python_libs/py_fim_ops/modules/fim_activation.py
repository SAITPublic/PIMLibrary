import torch
import torch.nn as nn
from torch.autograd import Function
from py_fim_ops import py_fim_activation


class FimActivationFunction(Function):
    @staticmethod
    def forward(ctx, inp):
        output = py_fim_activation(inp)

        return output

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


class FimActivation(nn.Module):
    """A nn.module wrapper for py_fim_activation function.
    """
    def __init__(self):
        super(FimActivation, self).__init__()

    def __repr__(self):
        return "Fim Activation Layer"

    def forward(self, inp):
        return FimActivationFunction.apply(inp)
