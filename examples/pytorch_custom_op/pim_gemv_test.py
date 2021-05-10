import unittest
import numpy as np
import torch
import py_pim_ops


class PyPimGemvTestConstant(unittest.TestCase):
    def test_gemv_constant(self):
        in_size = 800
        out_size = 3200

        np_inp = np.ones(shape=(1, in_size)).astype(np.float16)
        np_weight = np.ones(shape=(in_size, out_size)).astype(np.float16)
        true_output = np.ones(out_size).astype(np.float16) * in_size

        inp = torch.from_numpy(np_inp)
        weight = torch.from_numpy(np_weight)
        reorder = torch.tensor([1], dtype=torch.int32)

        result = py_pim_ops.py_pim_gemv(inp, weight, reorder)
        assert (result.numpy() == true_output.reshape(1, out_size)).all()

    def test_gemv_small(self):
        np_inp = np.ones(shape=(1, 64)).astype(np.float16)
        np_weight = np.ones(shape=(64, 32)).astype(np.float16)
        true_output = np.ones(32).astype(np.float16) * 64

        inp = torch.from_numpy(np_inp)
        weight = torch.from_numpy(np_weight)
        reorder = torch.tensor([1], dtype=torch.int32)

        result = py_pim_ops.py_pim_gemv(inp, weight, reorder)
        assert (result.numpy() == true_output.reshape(1, 32)).all()


class PyPimGemvTestRandom(unittest.TestCase):
    def test_gemv_random(self):
        minv = 0.5
        maxv = 1.0
        in_size = 800
        out_size = 3200
        inp = np.random.uniform(minv, maxv, size=[1,
                                                  in_size]).astype(np.float16)
        weight = np.random.uniform(minv, maxv,
                                   size=[in_size, out_size]).astype(np.float16)
        reorder = torch.tensor([1], dtype=torch.int32)

        true_output = np.matmul(inp, weight)
        result = py_pim_ops.py_pim_gemv(torch.from_numpy(inp),
                                        torch.from_numpy(weight), reorder)
        assert (np.allclose(result.numpy(), true_output, rtol=5e-1) == True)

    def test_gemv_coverage(self):
        minv = 0.5
        maxv = 1.0
        batches = [1, 4]
        sizes = [(128, 768), (256, 768), (384, 768), (128, 1024), (256, 1024),
                 (384, 1024), (800, 3200)]
        success = True
        failed_cases = []
        for batch in batches:
            for size in sizes:
                inp = np.random.uniform(minv, maxv,
                                        size=[batch,
                                              size[0]]).astype(np.float16)
                weight = np.random.uniform(minv, maxv,
                                           size=[size[0],
                                                 size[1]]).astype(np.float16)
                reorder = torch.tensor([1], dtype=torch.int32)

                true_output = np.matmul(inp, weight)
                result = py_pim_ops.py_pim_gemv(torch.from_numpy(inp),
                                                torch.from_numpy(weight),
                                                reorder)

                try:
                    assert (np.allclose(result.numpy(), true_output,
                                        rtol=5e-1) == True)
                except Exception as ex:
                    failed_cases.append([batch, size])
                    success = False

        if not success:
            print("Test cases failed!: " + str(failed_cases))
        assert (success == True)


if __name__ == '__main__':

    py_pim_ops.py_pim_init()
    unittest.main()
    py_pim_ops.py_pim_deinit()
