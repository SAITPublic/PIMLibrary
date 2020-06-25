from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tf_fim_ops

tf.debugging.set_log_device_placement(True)
testFilesPath = 'test_vectors/'


class FimAddTestConstant(tf.test.TestCase):
    def test_vector_vector(self):
        input0 = tf.constant([1., 2., 3., 4., 5.], dtype=np.float16)
        input1 = tf.constant([50., 5., 5., 5., 5.], dtype=np.float16)
        add = tf.constant([0], dtype=np.int32)
        with self.test_session():
            result = tf_fim_ops.fim_eltwise(input0, input1, add)
            self.assertAllEqual(result, [51., 7., 8., 9., 10.])

    def test_scalar_vector(self):
        input0 = tf.constant([20], dtype=np.float16)
        input1 = tf.constant(
            [[1., 2., 3., 4., 5.], [6., 7., 8., 9., 1.]], dtype=np.float16)
        add = tf.constant([0], dtype=np.int32)
        with self.test_session():
            result = tf_fim_ops.fim_eltwise(input0, input1, add)
            self.assertAllEqual(
                result, [[21., 22., 23., 24., 25.], [26., 27., 28., 29., 21]])

    def test_vector_scalar(self):
        input0 = tf.constant(
            [[1., 2., 3., 4., 5.], [6., 7., 8., 9., 1.]], dtype=np.float16)
        input1 = tf.constant([20], dtype=np.float16)
        add = tf.constant([0], dtype=np.int32)
        with self.test_session():
            result = tf_fim_ops.fim_eltwise(input0, input1, add)
            self.assertAllEqual(
                result, [[21., 22., 23., 24., 25.], [26., 27., 28., 29., 21]])

    def test_scalar_scalar(self):
        input0 = tf.constant([10], dtype=np.float16)
        input1 = tf.constant([100], dtype=np.float16)
        add = tf.constant([0], dtype=np.int32)
        with self.test_session():
            result = tf_fim_ops.fim_eltwise(input0, input1, add)
            self.assertAllEqual(result, [110])

    def test_2Dscalar_vector(self):
        input0 = tf.constant([[10]], dtype=np.float16)
        input1 = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float16)
        add = tf.constant([0], dtype=np.int32)
        with self.test_session():
            result = tf_fim_ops.fim_eltwise(input0, input1, add)
            self.assertAllEqual(result, [[11, 12, 13, 14], [15, 16, 17, 18]])


class FimAddTestRandom(tf.test.TestCase):
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
                            add = tf.constant([0], dtype=np.int32)

                            result_custom = tf_fim_ops.fim_eltwise(input0, input1, add)
                            result_math_add = tf.math.add(input0, input1)
                            try:
                                self.assertAllEqual(
                                    result_custom, result_math_add)
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
                            input1 = tf.constant([100], dtype=np.float16)

                            add = tf.constant([0], dtype=np.int32)

                            result_custom = tf_fim_ops.fim_eltwise(input0, input1, add)
                            result_math_add = tf.math.add(input0, input1)
                            try:
                                self.assertAllEqual(
                                    result_custom, result_math_add)
                            except Exception as ex:
                                failed_cases.append([b, c, w, h])
                                success = False
        if not success:
            print("Test cases failed!: " + str(failed_cases))
        self.assertEqual(success, True)


class FimAddTestFile(tf.test.TestCase):
    def test1(self):
        with self.test_session():
            input0 = np.fromfile(
                "test_vectors/load/elt_add/input0_256KB.dat",
                dtype=np.float16)
            input1 = np.fromfile(
                "test_vectors/load/elt_add/input1_256KB.dat",
                dtype=np.float16)

            add = tf.constant([0], dtype=np.int32)
            result = tf_fim_ops.fim_eltwise(input0, input1, add)
            golden = np.fromfile(
                "test_vectors/load/elt_add/output_256KB.dat",
                dtype=np.float16)
            self.assertAllEqual(result, golden)

    def test2(self):
        with self.test_session():
            input0 = np.fromfile(
                "test_vectors/load/elt_add/input0_512KB.dat",
                dtype=np.float16)
            input1 = np.fromfile(
                "test_vectors/load/elt_add/input1_512KB.dat",
                dtype=np.float16)

            add = tf.constant([0], dtype=np.int32)
            result = tf_fim_ops.fim_eltwise(input0, input1, add)
            golden = np.fromfile(
                "test_vectors/load/elt_add/output_512KB.dat",
                dtype=np.float16)
            self.assertAllEqual(result, golden)

    def test_scalar_vector(self):
        with self.test_session():
            input0 = np.fromfile(
                "test_vectors/load/sv_add/scalar_2B.dat",
                dtype=np.float16)
            input1 = np.fromfile(
                "test_vectors/load/sv_add/vector_256KB.dat",
                dtype=np.float16)

            add = tf.constant([0], dtype=np.int32)
            result = tf_fim_ops.fim_eltwise(input0, input1, add)
            golden = np.fromfile(
                "test_vectors/load/sv_add/output_256KB.dat",
                dtype=np.float16)
            self.assertAllEqual(result, golden)


if __name__ == '__main__':
    tf_fim_ops.fim_init()
    tf.test.main()
    tf_fim_ops.fim_deinit()
