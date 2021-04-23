import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Function
from py_fim_ops import py_fim_bn


class FimBNFunction(Function):
    @staticmethod
    def forward(ctx, inp, mean, var, offset, scale, epsilon):
        output = py_fim_bn(inp, mean, var, offset, scale, epsilon)

        return output

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


class FimBN(nn.Module):
    """A nn.module wrapper for py_fim_bn function.
    """
    def __init__(self, num_features, eps=0.001):
        super(FimBN, self).__init__()

        self.num_features = num_features
        self.epsilon = torch.tensor([eps], dtype=torch.double)

        #NOTE:  mean and var are set to zero and one respectively. Modify these values accordingly when creating an object.
        self.mean = torch.zeros(num_features, dtype=torch.float16)
        self.var = torch.ones(num_features, dtype=torch.float16)
        self.weight = Parameter(torch.HalfTensor(num_features))  #scale
        self.bias = Parameter(torch.HalfTensor(num_features))  #offset

    def __repr__(self):
        return "Fim BN Layer"

    def forward(self, inp):
        return FimBNFunction.apply(inp, self.mean, self.var, self.bias,
                                   self.weight, self.epsilon)
