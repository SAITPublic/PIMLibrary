import numpy as np
import torch
import torch.nn as nn

import py_fim_ops
from py_fim_ops import FimEltwise, FimBN, FimActivation, FimGemv


class SampleNetwork(nn.Module):
    def __init__(self):
        super(SampleNetwork, self).__init__()

        self.bn1 = FimBN(num_features=3, eps=1e-5)
        self.relu1 = FimActivation()

        self.bn2 = FimBN(num_features=3, eps=1e-5)
        self.relu2 = FimActivation()

        self.add = FimEltwise(operation=0)
        self.fc = FimGemv(in_features=2352, out_features=16, reorder=1)

    def forward(self, x1, x2):
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        x2 = self.bn2(x2)
        x2 = self.relu2(x2)

        x = self.add(x1, x2)
        x = torch.flatten(x, 1)
        output = self.fc(x)

        return output


if __name__ == '__main__':
    inp1 = torch.from_numpy(
        np.random.uniform(-500, 500, size=[2, 3, 28, 28]).astype(np.float16))
    inp2 = torch.from_numpy(
        np.random.uniform(-500, 500, size=[2, 3, 28, 28]).astype(np.float16))

    model = SampleNetwork()
    py_fim_ops.py_fim_init()
    output = model(inp1, inp2)
    py_fim_ops.py_fim_deinit()
    print("==>Output Tensor size is: ", output.size())
