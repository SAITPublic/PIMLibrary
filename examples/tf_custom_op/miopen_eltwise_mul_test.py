from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

try:
    from tf_fim_ops.python.ops.miopen_eltwise_ops import miopen_elt
except ImportError:
    from miopen_eltwise_ops import miopen_elt

tf.debugging.set_log_device_placement(True)
testFilesPath = '../test_vectors/'


class MIopenMulTestConstant(tf.test.TestCase):

    def test(self):
        with self.test_session():
            input0 = tf.constant([1., 2., 3., 4., 5.], dtype=np.float16)
            input1 = tf.constant([50., 5., 5., 5., 0.], dtype=np.float16)

            mul = tf.constant([1], dtype=np.int32)
            result = miopen_elt(input0, input1, mul)
            self.assertAllEqual(result, [50., 10., 15., 20., 0.])


class MIopenMulTestRandom(tf.test.TestCase):

    def test_4dim_4dim(self):
        batch_size = [1, 4, 8]
        channel = [1, 12]
        width = [128, 256, 384]
        height = [768, 1024]
        failed_cases = []
        success = True

        for b in batch_size:
            for c in channel:
                for w in width:
                    for h in height:
                        with self.test_session():
                            input0 = tf.random.uniform(shape=[b, c, w, h], minval=-500, maxval=500, dtype=np.float16)
                            input1 = tf.random.uniform(shape=[b, c, w, h], minval=-500, maxval=500, dtype=np.float16)
                            mul = tf.constant([1], dtype=np.int32)

                            result_custom = miopen_elt(input0, input1, mul)
                            result_math_multiply = tf.math.multiply(input0, input1)
                            try:
                                self.assertAllEqual(result_custom, result_math_multiply)
                            except Exception as ex:
                                failed_cases.append([b, c, w, h])
                                success = False
        print("Test cases failed!: " + str(failed_cases))
        self.assertEqual(success, True)

    def test_4im_scalar(self):
        batch_size = [1, 8]
        channel = [12]
        width = [128]
        height = [384]
        failed_cases = []
        success = True

        for b in batch_size:
            for c in channel:
                for w in width:
                    for h in height:
                        with self.test_session():
                            input0 = tf.random.uniform(shape=[b, c, w, h], minval=-500, maxval=500, dtype=np.float16)
                            input1 = tf.constant(np.random.randint(-500, 500), dtype=np.float16)

                            mul = tf.constant([1], dtype=np.int32)

                            result_custom = miopen_elt(input0, input1, mul)
                            result_math_multiply = tf.math.multiply(input0, input1)
                            try:
                                self.assertAllEqual(result_custom, result_math_multiply)
                            except Exception as ex:
                                failed_cases.append([b, c, w, h])
                                success = False
        print("Test cases failed!: " + str(failed_cases))
        self.assertEqual(success, True)


class MIopenMulTestFile(tf.test.TestCase):

    def test(self):
        with self.test_session():
            input0 = np.fromfile("../test_vectors/load/elt_mul/input0_128KB.dat", dtype=np.float16)
            input1 = np.fromfile("../test_vectors/load/elt_mul/input1_128KB.dat", dtype=np.float16)

            mul = tf.constant([1], dtype=np.int32)
            result = miopen_elt(input0, input1, mul)
            golden = np.fromfile("../test_vectors/load/elt_mul/output_128KB.dat", dtype = np.float16)
            self.assertAllEqual(result, golden)


if __name__ == '__main__':
  tf.test.main()
