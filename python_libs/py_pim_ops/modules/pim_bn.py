import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Function
from py_pim_ops import py_pim_bn


class PimBNFunction(Function):
    @staticmethod
    def forward(ctx, inp, mean, var, offset, scale, epsilon):
        output = py_pim_bn(inp, mean, var, offset, scale, epsilon)

        return output

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


class PimBN(nn.Module):
    """A nn.module wrapper for py_pim_bn function.
    """
    def __init__(self, num_features, eps=0.001):
        super(PimBN, self).__init__()

        self.num_features = num_features
        self.epsilon = torch.tensor([eps], dtype=torch.double)

        #NOTE:  mean and var are set to zero and one respectively. Modify these values accordingly when creating an object.
        self.mean = torch.zeros(num_features, dtype=torch.float16)
        self.var = torch.ones(num_features, dtype=torch.float16)
        self.weight = Parameter(torch.HalfTensor(num_features))  #scale
        self.bias = Parameter(torch.HalfTensor(num_features))  #offset

    def __repr__(self):
        return "Pim BN Layer"

    def forward(self, inp):
        return PimBNFunction.apply(inp, self.mean, self.var, self.bias,
                                   self.weight, self.epsilon)
