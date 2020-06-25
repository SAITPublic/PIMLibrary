from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tf_fim_ops

tf.debugging.set_log_device_placement(True)
testFilesPath = 'test_vectors/'


class FimActTestConstant(tf.test.TestCase):
    def test1(self):
        with self.test_session():
            inp = tf.constant([-3., 5., -13., -4., 9., 0.], dtype=np.float16)
            result = tf_fim_ops.fim_act(inp)
            self.assertAllEqual(result, [0., 5., 0., 0., 9., 0.])

    def test2(self):
        with self.test_session():
            inp = tf.constant(
                [[-5., -1., 0.], [2., -1., 0.]], dtype=np.float16)
            result = tf_fim_ops.fim_act(inp)
            self.assertAllEqual(result, [[0., 0., 0.], [2., 0., 0.]])


class FimActTestRandom(tf.test.TestCase):
    def test(self):
        batch_size = [1, 10]
        channel = [32, 128, 384]
        width = [3, 8, 73]
        height = [3, 8, 73]
        failed_cases = []
        success = True

        for b in batch_size:
            for c in channel:
                for w in width:
                    for h in height:
                        with self.test_session():
                            input0 = tf.random.uniform(
                                shape=[
                                    b,
                                    c,
                                    w,
                                    h],
                                minval=-
                                500,
                                maxval=500,
                                dtype=np.float16)
                            result_custom = tf_fim_ops.fim_act(input0)
                            result_tf_relu = tf.nn.relu(input0)
                            try:
                                self.assertAllEqual(
                                    result_custom, result_tf_relu)
                            except Exception as ex:
                                failed_cases.append([b, c, w, h])
                                success = False

        if not success:
            print("Test cases failed!: " + str(failed_cases))
        self.assertEqual(success, True)


class FimActTestFile(tf.test.TestCase):
    def test1(self):
        with self.test_session():
            inp = np.fromfile(
                "test_vectors/load/relu/input_256KB.dat",
                dtype=np.float16)
            result = tf_fim_ops.fim_act(inp)
            golden = np.fromfile(
                "test_vectors/load/relu/output_256KB.dat",
                dtype=np.float16)
            self.assertAllEqual(result, golden)

    def test2(self):
        with self.test_session():
            inp = np.fromfile(
                "test_vectors/load/relu/input_512KB.dat",
                dtype=np.float16)
            result = tf_fim_ops.fim_act(inp)
            golden = np.fromfile(
                "test_vectors/load/relu/output_512KB.dat",
                dtype=np.float16)
            self.assertAllEqual(result, golden)


if __name__ == '__main__':
    tf_fim_ops.fim_init()
    tf.test.main()
    tf_fim_ops.fim_deinit()
