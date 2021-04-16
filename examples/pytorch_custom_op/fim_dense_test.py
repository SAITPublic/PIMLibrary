import unittest
import numpy as np
import torch
import py_fim_ops

class Model(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(Model, self).__init__()
        self.fc = torch.nn.Linear(in_size, out_size)

    def forward(self, x):
        x = self.fc(x)
        return x

class PyDenseTest(unittest.TestCase):
    def testDense_1DRandom(self):
        in_batch = 2
        in_size = 16
        out_size = 4

        #This part creates a small model with an fc layer. Also,
        #we should keep all tensors on the GPU.
        a = torch.rand(size=(in_batch, 1, in_size), dtype=torch.float16)
        a = a.cuda()
        b = a.clone().detach()
        dense = Model(in_size, out_size)
        dense = dense.to("cuda:0")
        dense.half()

        pytorch_result = dense(a) #Pass the tensor to pytorch model and obtain output
        weights = dense.fc.weight #Weight copy to be used in FIM computation.
        print('Input shape',a.shape)
        print('Weight shape',weights.shape)
        print('Pytorch result shape',pytorch_result.shape)
        print('bias' , dense.fc.bias)
        has_bias = torch.Tensor([[1]]).to(torch.float16) #Bias flag indicating that bias component is there.
        has_bias = has_bias.cuda()
        bias = dense.fc.bias.clone().detach()
        fim_result = py_fim_ops.py_fim_dense(b, weights, has_bias, bias) #Obtain FIM output
        print("Result shape:", fim_result.shape)
        fim_result = fim_result.cpu().detach().numpy().reshape(1, 8)
        pytorch_result = pytorch_result.cpu().detach().numpy().reshape(1, 8)
        print("FIM Result:", fim_result)
        print("Pytorch Result:", pytorch_result)
        np.testing.assert_allclose(fim_result, pytorch_result, atol=0.01)


    def testDense_2DRandom(self):
        in_batch = 1
        in_size = 8
        out_size = 4
        
        #Test for passing a 2d tensor into FC layer.
        a = torch.rand(size=(in_batch, in_size*2 , in_size), dtype=torch.float16)
        a = a.cuda()
        b = a.clone().detach()
        dense = Model(in_size, out_size)
        dense = dense.to("cuda:0")
        dense.half()

        pytorch_result = dense(a) #Obtain pytorch output
        weights = dense.fc.weight
        print('Input shape',a.shape)
        print('Weight shape',weights.shape)
        print('Pytorch result shape',pytorch_result.shape)
        has_bias = torch.Tensor([[1]]).to(torch.float16)
        bias = dense.fc.bias.clone().detach()
        fim_result = py_fim_ops.py_fim_dense(a, weights, has_bias, bias) #Obtain FIM output
        print("Result shape:", fim_result.shape)
        fim_result = fim_result.cpu().detach().numpy().reshape(1, 64)
        pytorch_result = pytorch_result.cpu().detach().numpy().reshape(1, 64)
        print("FIM result", fim_result)
        print("Pytorch result:", pytorch_result)
        np.testing.assert_allclose(fim_result, pytorch_result, atol=0.01)


if __name__ == "__main__":
    py_fim_ops.py_fim_init()
    unittest.main()
    py_fim_ops.py_fim_deinit()
