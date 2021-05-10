from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import initializers
import numpy as np
import tf_pim_ops
import time
import timeit

tf.debugging.set_log_device_placement(True)
tf.keras.backend.set_floatx('float16')

eval_time = []

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
            result = tf_pim_ops.pim_dense(a, weights[0], weights[1], tf.constant([1]), tf.constant([1]))
            #print('Result',result,golden)
            self.assertAllClose(result, golden, atol=0.01)


    def testDense_2DRandom(self):
        eval_time.clear()
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
            result = tf_pim_ops.pim_dense(a, weights[0], weights[1], tf.constant([1]), tf.constant([1]))
            eval_time.append(timeit.timeit(lambda : tf_pim_ops.pim_dense(a, weights[0], weights[1], tf.constant([1]), tf.constant([1])), number = 10))
            print(eval_time)
            #result = tf_pim_ops.pim_dense(a, weights[0], weights[1], tf.constant([1]), tf.constant([1]))
            #print('Result',result,golden)
            self.assertAllClose(result, golden, atol=0.01)


    def testDense_2DRandom_layer(self):
        eval_time.clear()
        in_batch = 1
        in_size = 8
        out_size = 4

        a = tf.random.uniform(shape=(in_batch, in_size*2 , in_size), dtype=tf.float16)
        dense = tf.keras.layers.Dense(out_size, use_bias=True,  bias_initializer='glorot_uniform', dtype=tf.float16)
        pim_dense = PimDense(out_size, use_bias=True,  bias_initializer='zeros', dtype=tf.float16)

        #dummy run to init weights
        golden = dense(a)
        weights = dense.get_weights()
        pim_dense.set_weights(weights)

        with self.test_session():
            golden = dense(a)
            print('Input shape',a.shape)
            print('Weight shape',weights[0].shape)
            print('Golden shape',golden.shape)
            result = pim_dense(a)
            eval_time.append(timeit.timeit(lambda : pim_dense(a), number = 10))
            print(eval_time)
            #print('Result',result,golden)
            self.assertAllClose(result, golden, atol=0.01)


if __name__ == "__main__":
    tf_pim_ops.pim_init()
    tf.test.main()
    tf_pim_ops.pim_deinit()

