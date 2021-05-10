import unittest
import numpy as np
import torch
import py_pim_ops


class PyPimActTestConstant(unittest.TestCase):
    def test1(self):
        inp = torch.tensor([-3., 5., -13., -4., 9., 0.], dtype=torch.float16)

        result = py_pim_ops.py_pim_activation(inp)
        true_result = torch.tensor([0., 5., 0., 0., 9., 0.],
                                   dtype=torch.float16)
        assert (result.numpy() == true_result.numpy()).all()

    def test2(self):
        inp = torch.tensor([[-5., -1., 0.], [2., -1., 0.]],
                           dtype=torch.float16)

        result = py_pim_ops.py_pim_activation(inp)
        true_result = torch.tensor([[0., 0., 0.], [2., 0., 0.]],
                                   dtype=torch.float16)
        assert (result.numpy() == true_result.numpy()).all()


class PyPimActTestRandom(unittest.TestCase):
    def test(self):
        batch_size = [1, 10]
        channel = [32, 128, 384]
        height = [3, 8, 73]
        width = [3, 8, 73]

        failed_cases = []
        success = True

        for b in batch_size:
            for c in channel:
                for h in height:
                    for w in width:
                        min_val = -500
                        max_val = 500
                        inp = np.random.uniform(min_val,
                                                max_val,
                                                size=[b, c, h,
                                                      w]).astype(np.float16)
                        result = py_pim_ops.py_pim_activation(
                            torch.from_numpy(inp))
                        ## using numpy operations instead of torch.nn.ReLU because, most of the torch operations are not yet implemented for fp16.
                        true_result = np.maximum(inp, 0)
                        try:
                            assert (result.numpy() == true_result).all()
                        except Exception as ex:
                            failed_cases.append([b, c, h, w])
                            success = False

        if not success:
            print("Test cases failed!: " + str(failed_cases))
        assert (success == True)


class PyPimActTestFile(unittest.TestCase):
    def test1(self):
        inp = np.fromfile("test_vectors/load/relu/input_256KB.dat",
                          dtype=np.float16)

        result = py_pim_ops.py_pim_activation(torch.from_numpy(inp))
        golden = np.fromfile("test_vectors/load/relu/output_256KB.dat",
                             dtype=np.float16)
        assert (result.numpy() == golden).all()

    def test2(self):
        inp = np.fromfile("test_vectors/load/relu/input_512KB.dat",
                          dtype=np.float16)

        result = py_pim_ops.py_pim_activation(torch.from_numpy(inp))
        golden = np.fromfile("test_vectors/load/relu/output_512KB.dat",
                             dtype=np.float16)
        assert (result.numpy() == golden).all()


if __name__ == '__main__':

    py_pim_ops.py_pim_init()
    unittest.main()
    py_pim_ops.py_pim_deinit()
