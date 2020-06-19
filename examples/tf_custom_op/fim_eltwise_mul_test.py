from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import logging

from tensorflow.python.platform import test
try:
    from tf_fim_ops.python.ops.fim_eltwise_ops import fim_eltwise
except ImportError:
    from fim_eltwise_ops import fim_eltwise

try:
    from tf_fim_ops.python.ops.fim_init_ops import fim_init
    from tf_fim_ops.python.ops.fim_deinit_ops import fim_deinit
except ImportError:
    from fim_init_ops import fim_init
    from fim_deinit_ops import fim_deinit

tf.debugging.set_log_device_placement(True)
testFilesPath = 'test_vectors/'


class FimMulTestConstant(tf.test.TestCase):
    def test_vector_vector(self):
        input0 = tf.constant([1., 2., 3., 4., 5.], dtype=np.float16)
        input1 = tf.constant([50., 5., 5., 5., 0.], dtype=np.float16)
        mul = tf.constant([1], dtype=np.int32)
        with self.test_session():
            result = fim_eltwise(input0, input1, mul)
            self.assertAllEqual(result, [50., 10., 15., 20., 0.])

    def test_scalar_vector(self):
        input0 = tf.constant([20], dtype=np.float16)
        input1 = tf.constant(
            [[1., 2., 3., 4., 0.], [6., 7., 8., 9., 1.]], dtype=np.float16)
        mul = tf.constant([1], dtype=np.int32)
        with self.test_session():
            result = fim_eltwise(input0, input1, mul)
            self.assertAllEqual(result, [[20., 40., 60., 80., 0.], [
                                120., 140., 160., 180., 20.]])

    def test_vector_scalar(self):
        input0 = tf.constant(
            [[1., 2., 3., 4., 0.], [6., 7., 8., 9., 1.]], dtype=np.float16)
        input1 = tf.constant([20], dtype=np.float16)
        mul = tf.constant([1], dtype=np.int32)
        with self.test_session():
            result = fim_eltwise(input0, input1, mul)
            self.assertAllEqual(result, [[20., 40., 60., 80., 0.], [
                                120., 140., 160., 180., 20.]])

    def test_scalar_scalar(self):
        input0 = tf.constant([10], dtype=np.float16)
        input1 = tf.constant([100], dtype=np.float16)
        mul = tf.constant([1], dtype=np.int32)
        with self.test_session():
            result = fim_eltwise(input0, input1, mul)
            self.assertAllEqual(result, [1000])

    def test_2Dscalar_vector(self):
        input0 = tf.constant([[3]], dtype=np.float16)
        input1 = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float16)
        mul = tf.constant([1], dtype=np.int32)
        with self.test_session():
            result = fim_eltwise(input0, input1, mul)
            self.assertAllEqual(result, [[3, 6, 9, 12], [15, 18, 21, 24]])


class FimMulTestRandom(tf.test.TestCase):
    def test_4dim_4dim(self):
        batch_size = [1, 4, 8]
        channel = [1]
        width = [128, 256, 384]
        height = [768, 1024]
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
                            input1 = tf.random.uniform(
                                shape=[
                                    b,
                                    c,
                                    w,
                                    h],
                                minval=-
                                500,
                                maxval=500,
                                dtype=np.float16)
                            mul = tf.constant([1], dtype=np.int32)

                            result_custom = fim_eltwise(input0, input1, mul)
                            result_math_multiply = tf.math.multiply(
                                input0, input1)
                            try:
                                self.assertAllEqual(
                                    result_custom, result_math_multiply)
                            except Exception as ex:
                                failed_cases.append([b, c, w, h])
                                success = False
        if not success:
            print("Test cases failed!: " + str(failed_cases))
        self.assertEqual(success, True)

    def test_4dim_scalar(self):
        batch_size = [1, 8]
        channel = [1]
        width = [128]
        height = [768, 1024]
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
                            input1 = tf.constant([7], dtype=np.float16)

                            mul = tf.constant([1], dtype=np.int32)

                            result_custom = fim_eltwise(input0, input1, mul)
                            result_math_multiply = tf.math.multiply(
                                input0, input1)
                            try:
                                self.assertAllEqual(
                                    result_custom, result_math_multiply)
                            except Exception as ex:
                                failed_cases.append([b, c, w, h])
                                success = False
        if not success:
            print("Test cases failed!: " + str(failed_cases))
        self.assertEqual(success, True)


class FimMulTestFile(tf.test.TestCase):
    def test1(self):
        with self.test_session():
            input0 = np.fromfile(
                "test_vectors/load/elt_mul/input0_256KB.dat",
                dtype=np.float16)
            input1 = np.fromfile(
                "test_vectors/load/elt_mul/input1_256KB.dat",
                dtype=np.float16)

            mul = tf.constant([1], dtype=np.int32)
            result = fim_eltwise(input0, input1, mul)
            golden = np.fromfile(
                "test_vectors/load/elt_mul/output_256KB.dat",
                dtype=np.float16)
            self.assertAllEqual(result, golden)

    def test2(self):
        with self.test_session():
            input0 = np.fromfile(
                "test_vectors/load/elt_mul/input0_512KB.dat",
                dtype=np.float16)
            input1 = np.fromfile(
                "test_vectors/load/elt_mul/input1_512KB.dat",
                dtype=np.float16)

            mul = tf.constant([1], dtype=np.int32)
            result = fim_eltwise(input0, input1, mul)
            golden = np.fromfile(
                "test_vectors/load/elt_mul/output_512KB.dat",
                dtype=np.float16)
            self.assertAllEqual(result, golden)

    def test_scalar_vector(self):
        with self.test_session():
            input0 = np.fromfile(
                "test_vectors/load/sv_mul/scalar_2B.dat",
                dtype=np.float16)
            input1 = np.fromfile(
                "test_vectors/load/sv_mul/vector_256KB.dat",
                dtype=np.float16)

            mul = tf.constant([1], dtype=np.int32)
            result = fim_eltwise(input0, input1, mul)
            golden = np.fromfile(
                "test_vectors/load/sv_mul/output_256KB.dat",
                dtype=np.float16)
            self.assertAllEqual(result, golden)


if __name__ == '__main__':
    fim_init("RT_TYPE_HIP", "FIM_FP16")
    tf.test.main()
    fim_deinit()
