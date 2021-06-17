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
        gpu0 = torch.device(0)
        BATCH = 1
        CH = 1
        WIDTH = 131072
        HEIGHT = 1

        input0 = np.fromfile(
            "test_vectors/load/bn/nr_input_256KB.dat",
            dtype=np.float16)
        inp = torch.from_numpy(input0)

        beta = np.fromfile(
            "test_vectors/load/bn/nr_beta_256KB.dat",
            dtype=np.float16)
        beta = beta[0:CH]
        beta = torch.from_numpy(beta)

        gamma = np.fromfile(
            "test_vectors/load/bn/nr_gamma_256KB.dat",
            dtype=np.float16)
        gamma = gamma[0:CH]
        gamma = torch.from_numpy(gamma)

        mean = np.fromfile(
            "test_vectors/load/bn/nr_mean_256KB.dat",
            dtype=np.float16)
        mean = mean[0:CH]
        mean = torch.from_numpy(mean)

        var = np.fromfile(
            "test_vectors/load/bn/nr_variance_256KB.dat",
            dtype=np.float16)
        var = var[0:CH]
        var = torch.from_numpy(var)
        epsilon = torch.tensor([1e-5], dtype=torch.double)

        golden = np.fromfile(
            "test_vectors/load/bn/nr_output_256KB.dat",
            dtype=np.float16)
        golden = torch.from_numpy(golden)

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
