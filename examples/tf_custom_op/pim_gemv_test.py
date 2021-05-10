from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tf_pim_ops

tf.debugging.set_log_device_placement(True)
testFilesPath = '../test_vectors/'


class GemvTest(tf.test.TestCase):
    def testGemvConstant(self):
        in_size = 800
        out_size = 3200
        an = np.ones(shape=(1, in_size))
        bn = np.ones(shape=(in_size, out_size))
        a = tf.constant(an, dtype=np.float16)
        b = tf.constant(bn, dtype=np.float16)
        o = np.ones(out_size) * in_size

        with self.test_session():
            #result = tf.linalg.matmul(a,b)
            result = tf_pim_ops.pim_gemv(a, b, tf.constant([1]))
            self.assertAllEqual(result, o.reshape(1, out_size))

    def testGemvRandom(self):
        minv = 0.5
        maxv = 1.0
        in_size = 800
        out_size = 3200
        a = tf.random.uniform(
            shape=(
                1,
                in_size),
            minval=minv,
            maxval=maxv,
            dtype=tf.dtypes.float16)
        b = tf.random.uniform(
            shape=(
                in_size,
                out_size),
            minval=minv,
            maxval=maxv,
            dtype=tf.dtypes.float16)
        with self.test_session():
            golden = tf.linalg.matmul(a, b)
            result = tf_pim_ops.pim_gemv(a, b, tf.constant([1]))
            self.assertAllClose(result, golden, rtol=5e-1)

    # Todo,support batchsize when implement in pim

    def testGemvCoverage(self):
        with self.test_session():
            minv = 0.5
            maxv = 1.0 
            batches = [1,4]
            sizes = [
                (128, 768),
                (256, 768),
                (384, 768),
                (128, 1024),
                (256, 1024),
                (384, 1024),
                (800, 3200)
            ]
            success = True
            failed_cases = []
            for batch in batches:
              for size in sizes:
                a = tf.random.uniform(
                    shape=(
                        batch,
                        size[0]),
                    minval=minv,
                    maxval=maxv,
                    dtype=tf.dtypes.float16)
                b = tf.random.uniform(
                    shape=(
                        size[0],
                        size[1]),
                    minval=minv,
                    maxval=maxv,
                    dtype=tf.dtypes.float16)
                golden = tf.linalg.matmul(a, b)
                result = tf_pim_ops.pim_gemv(a, b, tf.constant([1]))
                try:
                    self.assertAllClose(result, golden, rtol=5e-1)
                except Exception as ex:
                    failed_cases.append([batch,size])
                    success = False

            if not success:
                print("Test cases failed!: " + str(failed_cases))
            self.assertEqual(success, True)

    def testGemvGolden(self):
        in_size = 256
        out_size = 4096
        input0 = np.fromfile(
            testFilesPath +
            "load/gemv/input_256x1.dat",
            dtype=np.float16)
        input1 = np.fromfile(
            testFilesPath +
            "load/gemv/weight_256x4096.dat",
            dtype=np.float16)
        output0 = np.fromfile(
            testFilesPath +
            "load/gemv/output_4096x1.dat",
            dtype=np.float16)

        t_input0 = tf.convert_to_tensor(input0, np.float16)
        t_input1 = tf.convert_to_tensor(input1, np.float16)
        t_output0 = tf.convert_to_tensor(output0, np.float16)

        a = tf.reshape(t_input0, [1, in_size])
        # w = tf.reshape(t_input1, [in_size, out_size])
        w = tf.reshape(t_input1, [out_size, in_size])
        w = tf.transpose(w)
        o = tf.reshape(t_output0, [1, out_size])

        with self.test_session():
            result = tf_pim_ops.pim_gemv(a, w, tf.constant([1]))
            self.assertAllEqual(result, o)

    def testGemvGoldenBatch(self):
        in_size = 1024
        out_size = 4096
        batch_size = 4
        input0 = np.fromfile(
            testFilesPath +
            "load/gemv/gemv_batch_input_4x1024.dat",
            dtype=np.float16)
        input1 = np.fromfile(
            testFilesPath +
            "load/gemv/gemv_batch_weight_1024x4096.dat",
            dtype=np.float16)
        output0 = np.fromfile(
            testFilesPath +
            "load/gemv/gemv_batch_output_4x4096.dat",
            dtype=np.float16)

        t_input0 = tf.convert_to_tensor(input0, np.float16)
        t_input1 = tf.convert_to_tensor(input1, np.float16)
        t_output0 = tf.convert_to_tensor(output0, np.float16)

        a = tf.reshape(t_input0, [batch_size, in_size])
        # w = tf.reshape(t_input1, [in_size, out_size])
        w = tf.reshape(t_input1, [out_size, in_size])
        w = tf.transpose(w)
        o = tf.reshape(t_output0, [batch_size, out_size])

        with self.test_session():
            result = tf_pim_ops.pim_gemv(a, w, tf.constant([1]))
            self.assertAllEqual(result, o)

    def testGemvGoldenPad(self):
        in_size = 1024
        out_size = 4096
        input0 = np.fromfile(
            testFilesPath +
            "load/gemv/gemv_input_1024x4096.dat",
            dtype=np.float16)
        input1 = np.fromfile(
            testFilesPath +
            "load/gemv/gemv_weight_1024x4096.dat",
            dtype=np.float16)
        output0 = np.fromfile(
            testFilesPath +
            "load/gemv/gemv_output_1024x4096.dat",
            dtype=np.float16)

        t_input0 = tf.convert_to_tensor(input0, np.float16)
        t_input1 = tf.convert_to_tensor(input1, np.float16)
        t_output0 = tf.convert_to_tensor(output0, np.float16)

        a = tf.reshape(t_input0, [1, in_size])
        # w = tf.reshape(t_input1, [in_size, out_size])
        w = tf.reshape(t_input1, [out_size, in_size])
        w = tf.transpose(w)
        o = tf.reshape(t_output0, [1, out_size])

        with self.test_session():
            result = tf_pim_ops.pim_gemv(a, w, tf.constant([1]))
            self.assertAllEqual(result, o)

    def testGemvGoldenWeightReordered(self):
        in_size = 256
        out_size = 4096
        input0 = np.fromfile(
            testFilesPath +
            "load/gemv/input_256x1.dat",
            dtype=np.float16)
        input1 = np.fromfile(
            testFilesPath +
            "load/gemv/reordered_weight_256x4096.dat",
            dtype=np.float16)
        output0 = np.fromfile(
            testFilesPath +
            "load/gemv/output_4096x1.dat",
            dtype=np.float16)

        t_input0 = tf.convert_to_tensor(input0, np.float16)
        t_input1 = tf.convert_to_tensor(input1, np.float16)
        t_output0 = tf.convert_to_tensor(output0, np.float16)

        a = tf.reshape(t_input0, [1, in_size])
        # w = tf.reshape(t_input1, [in_size, out_size])
        w = tf.reshape(t_input1, [out_size, in_size])
        w = tf.transpose(w)
        o = tf.reshape(t_output0, [1, out_size])

        with self.test_session():
            result = tf_pim_ops.pim_gemv(a, w, tf.constant([0]))
            self.assertAllEqual(result, o)

    def testGemvSmall(self):
        an = np.ones(shape=(1, 64))
        bn = np.ones(shape=(64, 32))

        an = an * 1
        bn = bn * 1
        a = tf.constant(an, dtype=np.float16)
        b = tf.constant(bn, dtype=np.float16)
        with self.test_session():
            result = tf_pim_ops.pim_gemv(a, b, tf.constant([1]))
            result = tf.reshape(result, [32])
            self.assertAllEqual(result, [64.] * 32)


if __name__ == "__main__":
    tf_pim_ops.pim_init()
    tf.test.main()
    tf_pim_ops.pim_deinit()

