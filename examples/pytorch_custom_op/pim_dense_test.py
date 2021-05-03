import unittest
import numpy as np
import torch
import py_pim_ops

class Model(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(Model, self).__init__()
        self.fc = torch.nn.Linear(in_size, out_size ,bias=True)

    def forward(self, x):
        x = self.fc(x)
        return x

class PyDenseTest(unittest.TestCase):
    def testDense_1DRandom(self):
        py_pim_ops.py_pim_init()
        in_size = 8
        out_size = 16

        device = torch.device('cuda')
        #This part creates a small model with an fc layer. Also,
        #we should keep all tensors on the GPU.
        input = torch.ones(size=(in_size , in_size), dtype=torch.float16)
        input = input.to(device)
        dense = Model(in_size, out_size)
        dense = dense.to(device)
        dense.half()

        pytorch_result = dense(input) #Pass the tensor to pytorch model and obtain output
        weights = dense.fc.weight #Weight copy to be used in PIM computation.
        weights_t = weights.clone().detach()
        weights_t = torch.transpose(weights_t,0,1).contiguous()

        has_bias = torch.Tensor([[1]]).to(torch.float16) #Bias flag indicating that bias component is there.
        has_bias = has_bias.cuda()
        bias = dense.fc.bias.clone().detach()
        pim_result = py_pim_ops.py_pim_dense(input, weights_t, has_bias, bias) #Obtain PIM output
        #print("PIM Result:", pim_result)
        #print("Pytorch Result:", pytorch_result)
        self.assertTrue(torch.allclose(pim_result, pytorch_result, atol=0.01))
        py_pim_ops.py_pim_deinit()

    def testDense_2DRandom(self):
        py_pim_ops.py_pim_init()
        in_batch = 2
        in_size = 4
        out_size = 32
        
        device = torch.device('cuda')
        #Test for passing a 2d tensor into FC layer.
        input = torch.ones(size=(in_batch ,in_size , in_size), dtype=torch.float16)
        input = input.to(device)
        dense1 = Model(in_size, out_size)
        dense1 = dense1.to(device)
        dense1.half()

        pytorch_result = dense1(input) #Pass the tensor to pytorch model and obtain output
        weights = dense1.fc.weight #Weight copy to be used in PIM computation.
        weights_t = weights.clone().detach()
        weights_t = torch.transpose(weights_t,0,1).contiguous()

        has_bias = torch.Tensor([[1]]).to(torch.float16) #Bias flag indicating that bias component is there.
        has_bias = has_bias.cuda()
        bias = dense1.fc.bias.clone().detach()
        pim_result = py_pim_ops.py_pim_dense(input, weights_t, has_bias, bias) #Obtain PIM output
        #print("PIM Result:", pim_result)
        #print("Pytorch Result:", pytorch_result)
        self.assertTrue(torch.allclose(pim_result, pytorch_result, atol=0.01))
        py_pim_ops.py_pim_deinit()

if __name__ == "__main__":
    #note we move this to individual test cases since we are using same model , so bundle is same.
    #py_pim_ops.py_pim_init()
    unittest.main()
    #py_pim_ops.py_pim_deinit()
