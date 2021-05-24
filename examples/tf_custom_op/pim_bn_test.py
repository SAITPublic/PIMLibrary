from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tf_pim_ops

tf.debugging.set_log_device_placement(True)


class PimBn4DGolden(tf.test.TestCase):
    def test(self):
        with self.test_session():
            BATCH = 1
            CH = 1
            WIDTH = 131072
            HEIGHT = 1

            input0 = np.fromfile(
                "test_vectors/load/bn/nr_input_256KB.dat",
                dtype=np.float16)
            t_input0 = tf.convert_to_tensor(input0, np.float16)

            beta = np.fromfile(
                "test_vectors/load/bn/nr_beta_256KB.dat",
                dtype=np.float16)
            beta = beta[0:CH]
            t_beta = tf.convert_to_tensor(beta, np.float16)

            gamma = np.fromfile(
                "test_vectors/load/bn/nr_gamma_256KB.dat",
                dtype=np.float16)
            gamma = gamma[0:CH]
            t_gamma = tf.convert_to_tensor(gamma, np.float16)

            mean = np.fromfile(
                "test_vectors/load/bn/nr_mean_256KB.dat",
                dtype=np.float16)
            mean = mean[0:CH]
            t_mean = tf.convert_to_tensor(mean, np.float16)

            var = np.fromfile(
                "test_vectors/load/bn/nr_variance_256KB.dat",
                dtype=np.float16)
            var = var[0:CH]
            t_var = tf.convert_to_tensor(var, np.float16)

            epsilon = 1e-5
            golden = np.fromfile(
                "test_vectors/load/bn/nr_output_256KB.dat",
                dtype=np.float16)
            t_golden = tf.convert_to_tensor(golden, np.float16)

            t_input0 = tf.reshape(t_input0, [BATCH, CH, HEIGHT, WIDTH])
            t_golden = tf.reshape(t_golden, [BATCH, CH, HEIGHT, WIDTH])
            t_mean = tf.reshape(t_mean, [1, CH, 1, 1])
            t_var = tf.reshape(t_var, [1, CH, 1, 1])
            t_gamma = tf.reshape(t_gamma, [1, CH, 1, 1])
            t_beta = tf.reshape(t_beta, [1, CH, 1, 1])

            #result = tf.nn.batch_normalization(t_input0, t_mean, t_var, t_beta, t_gamma, epsilon)
            result = tf_pim_ops.pim_bn(t_input0, t_mean, t_var, t_beta, gamma, epsilon)
            self.assertAllClose(result, t_golden, atol=5e-3)


class PimBn4DOpc(tf.test.TestCase):

    def test(self):
        with self.test_session():
            sizes = [
                #(1, 1, 128, 768),
                #(1, 1, 256, 768),
                #(1, 1, 384, 768),
                (4, 1, 128, 768),
                (4, 1, 256, 768),
                (4, 1, 384, 768),
                (8, 1, 128, 768),
                (8, 1, 256, 768),
                (8, 1, 384, 768),
                (1, 1, 128, 1024),
                (1, 1, 256, 1024),
                (1, 1, 384, 1024),
                (4, 1, 128, 1024),
                (4, 1, 256, 1024),
                (4, 1, 384, 1024),
                (8, 1, 128, 1024),
                (8, 1, 256, 1024),
                (8, 1, 384, 1024),
                (1, 1, 1 , 131072),
                (1, 1, 1 , 131072 * 2),
                (1, 1, 1 , 131072 * 4),
                (1, 1, 1 , 131072 * 8),
                (1, 1, 1 , 131072 * 16),
                (1, 1, 1 , 131072 * 32),
                (1, 1, 1 , 131072 * 64),
                (1, 1, 1 , 131072 * 128),
            ]
            success = True
            failed_cases = []
            for size in sizes:
                input0 = tf.random.uniform(
                    shape=size, minval=0, maxval=8, dtype=np.float16)
                input0 = tf.reshape(input0, size)
                t_input0 = tf.convert_to_tensor(input0, np.float16)

                var_epsilon = 1e-5
                mean = tf.reduce_mean(t_input0, axis=[0, 2, 3])
                var = tf.math.reduce_std(t_input0, axis=[0, 2, 3])

                mean = tf.zeros(mean.shape, dtype=tf.dtypes.float16)
                var = tf.ones(var.shape, dtype=tf.dtypes.float16)

                offset = tf.zeros(mean.shape, dtype=tf.dtypes.float16)
                scale = tf.ones(mean.shape, dtype=tf.dtypes.float16)

                result = tf_pim_ops.pim_bn(
                    t_input0, mean, var, offset, scale, var_epsilon)
                tf_golden = tf.nn.batch_normalization(
                    t_input0, mean, var, offset, scale, var_epsilon)

                try:
                    self.assertAllClose(result, tf_golden, atol=1e-2)
                except Exception as ex:
                    failed_cases.append(size)
                    success = False

            if not success:
                print("Test cases failed!: " + str(failed_cases))
            self.assertEqual(success, True)


class PimBnRandom4D(tf.test.TestCase):

    def test(self):
        with self.test_session():
            input0 = tf.random.uniform(
                shape=[
                    2,
                    64,
                    1,
                    1024],
                minval=0,
                maxval=64,
                dtype=np.float16)
            t_input0 = tf.convert_to_tensor(input0, np.float16)

            beta = tf.zeros([1, 64, 1, 1], dtype=tf.dtypes.float16)
            gamma = tf.ones([1, 64, 1, 1], dtype=tf.dtypes.float16)
            mean = tf.zeros([1, 64, 1, 1], dtype=tf.dtypes.float16)
            var = tf.ones([1, 64, 1, 1], dtype=tf.dtypes.float16)

            var_epsilon = 1e-5
            bn = tf_pim_ops.pim_bn(t_input0, mean, var, beta, gamma, var_epsilon)
            #bn_tf = tf.nn.batch_normalization(t_input0, mean, var, beta, gamma, var_epsilon)
            #self.assertAllEqual(bn, input0)
            self.assertAllClose(bn, t_input0, rtol=5e-3)


if __name__ == '__main__':
    tf_pim_ops.pim_init()
    tf.test.main()
    tf_pim_ops.pim_deinit()
