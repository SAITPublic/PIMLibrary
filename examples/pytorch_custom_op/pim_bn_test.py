import unittest
import numpy as np
import torch
import py_pim_ops


class PyPimBNTestRandom(unittest.TestCase):
    def test(self):
        min_val = 0
        max_val = 64
        shape = [2, 64, 1, 1024]
        np_input = np.random.uniform(min_val, max_val,
                                     size=shape).astype(np.float16)

        mean = torch.zeros([1, 64, 1, 1], dtype=torch.float16)
        var = torch.ones([1, 64, 1, 1], dtype=torch.float16)
        beta = torch.zeros([1, 64, 1, 1], dtype=torch.float16)  # offset
        gamma = torch.ones([1, 64, 1, 1], dtype=torch.float16)  # scale
        eps = 0.0

        var_epsilon = torch.tensor([eps], dtype=torch.double)
        result = py_pim_ops.py_pim_bn(torch.from_numpy(np_input), mean, var,
                                      beta, gamma, var_epsilon)
        assert (result.numpy() == np_input).all()


class PyPimBNTestFile(unittest.TestCase):
    def test(self):
        BATCH = 2
        CH = 64
        HEIGHT = 1
        WIDTH = 1024

        inp = torch.from_numpy(
            np.fromfile("test_vectors/load/bn/input_256KB.dat",
                        dtype=np.float16))

        mean = torch.from_numpy(
            np.fromfile("test_vectors/load/bn/mean_128B.dat",
                        dtype=np.float16))

        var = torch.from_numpy(
            np.fromfile("test_vectors/load/bn/variance_128B.dat",
                        dtype=np.float16))

        beta = torch.from_numpy(
            np.fromfile("test_vectors/load/bn/beta_128B.dat",
                        dtype=np.float16))

        gamma = torch.from_numpy(
            np.fromfile("test_vectors/load/bn/gamma_128B.dat",
                        dtype=np.float16))

        epsilon = torch.tensor([1e-5], dtype=torch.double)

        golden = np.fromfile("test_vectors/load/bn/output_256KB.dat",
                             dtype=np.float16)
        golden = golden.reshape(BATCH, CH, HEIGHT, WIDTH)

        inp = torch.reshape(inp, [BATCH, CH, HEIGHT, WIDTH])
        mean = torch.reshape(mean, [1, CH, 1, 1])
        var = torch.reshape(var, [1, CH, 1, 1])
        beta = torch.reshape(beta, [1, CH, 1, 1])
        gamma = torch.reshape(gamma, [1, CH, 1, 1])

        result = py_pim_ops.py_pim_bn(inp, mean, var, beta, gamma, epsilon)
        assert (np.allclose(result.numpy(), golden, rtol=5e-1) == True)


if __name__ == '__main__':

    py_pim_ops.py_pim_init()
    unittest.main()
    py_pim_ops.py_pim_deinit()
