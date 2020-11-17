from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import initializers
import numpy as np
import tf_fim_ops


tf.debugging.set_log_device_placement(True)

class DenseTest(tf.test.TestCase):
    def testDense_1DRandom(self):
        in_batch = 2
        in_size = 16
        out_size = 4

        a = tf.random.uniform(shape=(in_batch, in_size), dtype=tf.float16)
        b = tf.random.uniform(shape=(in_size, out_size), dtype=tf.float16)
        dense = tf.keras.layers.Dense(out_size, use_bias=True,  bias_initializer=initializers.Constant(0.1), dtype=tf.float16)

        with self.test_session():
            golden = dense(a)
            weights = dense.get_weights()
            print('Input shape',a.shape)
            print('Weight shape',weights[0].shape)
            print('Golden shape',golden.shape)
            print('bias' , weights[1])
            tf_fim_ops.fim_init()
            result = tf_fim_ops.fim_dense(a, weights[0], weights[1], tf.constant([1]), tf.constant([1]))
            tf_fim_ops.fim_deinit()
            #print('Result',result,golden)
            self.assertAllClose(result, golden, atol=0.01)


    def testDense_2DRandom(self):
        in_batch = 1
        in_size = 8
        out_size = 4

        a = tf.random.uniform(shape=(in_batch, in_size*2 , in_size), dtype=tf.float16)
        dense = tf.keras.layers.Dense(out_size, use_bias=True,  bias_initializer='glorot_uniform', dtype=tf.float16)

        with self.test_session():
            golden = dense(a)
            weights = dense.get_weights()
            print('Input shape',a.shape)
            print('Weight shape',weights[0].shape)
            print('Golden shape',golden.shape)
            tf_fim_ops.fim_init()
            result = tf_fim_ops.fim_dense(a, weights[0], weights[1], tf.constant([1]), tf.constant([1]))
            #result = tf_fim_ops.fim_dense(a, weights[0], weights[1], tf.constant([1]), tf.constant([1]))
            tf_fim_ops.fim_deinit()
            print('Result',result,golden)
            self.assertAllClose(result, golden, atol=0.01)


if __name__ == "__main__":
    tf.test.main()

