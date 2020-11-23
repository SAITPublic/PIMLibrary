from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import initializers
import numpy as np
import tf_fim_ops
import time
import timeit

tf.debugging.set_log_device_placement(True)
tf.keras.backend.set_floatx('float16')

eval_time = []
class FimDenseLayer(tf.keras.layers.Layer):
    def __init__(self, weight, bias, dtype=tf.float16):
        super(FimDenseLayer, self).__init__()
        self.kernel = weight
        self.bias = bias

    def build(self, input_shape):
        pass

    def call(self, input):
      return tf_fim_ops.fim_dense(input, self.kernel, self.bias, tf.constant([1]), tf.constant([1]))


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
            result = tf_fim_ops.fim_dense(a, weights[0], weights[1], tf.constant([1]), tf.constant([1]))
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
            result = tf_fim_ops.fim_dense(a, weights[0], weights[1], tf.constant([1]), tf.constant([1]))
            eval_time.append(timeit.timeit(lambda : tf_fim_ops.fim_dense(a, weights[0], weights[1], tf.constant([1]), tf.constant([1])), number = 10))
            print(eval_time)
            #result = tf_fim_ops.fim_dense(a, weights[0], weights[1], tf.constant([1]), tf.constant([1]))
            #print('Result',result,golden)
            self.assertAllClose(result, golden, atol=0.01)

    def testDense_2DRandom_layer(self):
        eval_time.clear()
        in_batch = 1
        in_size = 8
        out_size = 4

        a = tf.random.uniform(shape=(in_batch, in_size*2 , in_size), dtype=tf.float16)
        dense = tf.keras.layers.Dense(out_size, use_bias=True,  bias_initializer='glorot_uniform', dtype=tf.float16)

        #dummy run to init weights
        golden = dense(a)
        weights = dense.get_weights()
        fim_dense_layer = FimDenseLayer(weights[0],weights[1],dtype=tf.float16)

        with self.test_session():
            golden = dense(a)
            weights = dense.get_weights()
            print('Input shape',a.shape)
            print('Weight shape',weights[0].shape)
            print('Golden shape',golden.shape)
            result = fim_dense_layer(a)
            eval_time.append(timeit.timeit(lambda : fim_dense_layer(a), number = 10))
            print(eval_time)
            #print('Result',result,golden)
            self.assertAllClose(result, golden, atol=0.01)


if __name__ == "__main__":
    tf_fim_ops.fim_init()
    tf.test.main()
    tf_fim_ops.fim_deinit()

