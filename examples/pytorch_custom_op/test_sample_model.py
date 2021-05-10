import numpy as np
import torch
import torch.nn as nn

import py_pim_ops
from py_pim_ops import PimEltwise, PimBN, PimActivation, PimGemv


class SampleNetwork(nn.Module):
    def __init__(self):
        super(SampleNetwork, self).__init__()

        self.bn1 = PimBN(num_features=3, eps=1e-5)
        self.relu1 = PimActivation()

        self.bn2 = PimBN(num_features=3, eps=1e-5)
        self.relu2 = PimActivation()

        self.add = PimEltwise(operation=0)
        self.fc = PimGemv(in_features=2352, out_features=16, reorder=1)

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
    py_pim_ops.py_pim_init()
    output = model(inp1, inp2)
    py_pim_ops.py_pim_deinit()
    print("==>Output Tensor size is: ", output.size())
