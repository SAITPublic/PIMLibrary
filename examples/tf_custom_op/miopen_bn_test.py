from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import math
import tf_fim_ops

tf.debugging.set_log_device_placement(True)


class MIopenBn4DGolden(tf.test.TestCase):

  def test(self):
    with self.test_session():

        BATCH = 2;
        CH = 64
        WIDTH = 1024;
        HEIGHT = 1;

        input0  =  np.fromfile("../test_vectors/load/bn/input_256KB.dat", dtype = np.float16)
        t_input0   = tf.convert_to_tensor(input0, np.float16)

        beta  =  np.fromfile("../test_vectors/load/bn/beta_128B.dat", dtype = np.float16)
        t_beta  = tf.convert_to_tensor(beta, np.float16)

        gamma  =  np.fromfile("../test_vectors/load/bn/gamma_128B.dat", dtype = np.float16)
        t_gamma  = tf.convert_to_tensor(gamma, np.float16)

        scale  =  np.fromfile("../test_vectors/load/bn/scale_128B.dat", dtype = np.float16)
        t_scale  = tf.convert_to_tensor(scale, np.float16)

        shift  =  np.fromfile("../test_vectors/load/bn/shift_128B.dat", dtype = np.float16)
        t_shift   = tf.convert_to_tensor(shift, np.float16)

        epsilon =1e-3
        var = np.reciprocal(scale)
        var = np.multiply(var,var)
        var = var - epsilon
        t_var  = tf.convert_to_tensor(var, np.float16)

        mean = -np.divide(shift,scale)
        t_mean  = tf.convert_to_tensor(mean, np.float16)

        golden  =  np.fromfile("../test_vectors/load/bn/output_256KB.dat", dtype = np.float16)
        t_golden   = tf.convert_to_tensor(golden, np.float16)

        t_input0 = tf.reshape(t_input0,[BATCH,CH,WIDTH,HEIGHT])
        t_golden = tf.reshape(t_golden,[BATCH,CH,WIDTH,HEIGHT])
        t_mean= tf.reshape(t_mean,[1,CH,1,1])
        t_var = tf.reshape(t_var,[1,CH,1,1])
        t_beta = tf.reshape(t_beta,[1,CH,1,1])
        t_gamma = tf.reshape(t_gamma,[1,CH,1,1])

        #result = tf.nn.batch_normalization(t_input0, t_mean, t_var, t_beta, t_gamma, epsilon)
        result = tf_fim_ops.miopen_bn(t_input0, t_mean, t_var, t_beta, t_gamma, epsilon)
        self.assertAllClose(result, t_golden, atol=5e-3)

class MIopenBn1D(tf.test.TestCase):

  def _test(self):
      with self.test_session():
        sizes = [15,64,512,8096,32565,65535]

        for size in sizes:
          input0 = tf.random.uniform(shape=[size], minval=0, maxval=64, dtype = np.float16)
          t_input0   = tf.convert_to_tensor(input0, np.float16)

          mean = 0.
          var = 1.
          offset = 0.
          scale = 1.
          var_epsilon = 1e-5

          bn = tf.nn.batch_normalization(t_input0, mean, var, offset, scale, var_epsilon)
          self.assertAllEqual(bn,input0)


class MIopenBnOffset1D(tf.test.TestCase):

  def _test(self):
      with self.test_session():
        sizes = [15,64,512,8096,32565,65535]

        for size in sizes:
          input0 = tf.random.uniform(shape=[size], minval=0, maxval=64, dtype = np.float16)
          t_input0 = tf.convert_to_tensor(input0, np.float16)

          mean = 0.
          var = 1.
          offset = 1.
          scale = 1.
          var_epsilon = 1e-5

          bn = tf.nn.batch_normalization(t_input0, mean, var, offset, scale, var_epsilon)
          self.assertAllEqual(bn,offset + input0)


class MIopenBnScale1D(tf.test.TestCase):

  def _test(self):
      with self.test_session():
        sizes = [15,64,512,8096,32565,65535]

        for size in sizes:
          input0 = tf.random.uniform(shape=[size], minval=0, maxval=64, dtype = np.float16)
          t_input0 = tf.convert_to_tensor(input0, np.float16)

          mean = 0.
          var = 1.
          offset = 0.
          scale = 2.
          var_epsilon = 1e-6

          bn = tf.nn.batch_normalization(t_input0, mean, var, offset, scale, var_epsilon)
          self.assertAllEqual(bn,scale * input0)

class MIopenBnVariance1D(tf.test.TestCase):

  def _test(self):
      with self.test_session():
        sizes = [15,64,512,8096,32565,65535]

        for size in sizes:
          input0 = tf.random.uniform(shape=[size], minval=0, maxval=64, dtype = np.float16)
          t_input0 = tf.convert_to_tensor(input0, np.float16)

          mean = 0.
          var = 0.1
          offset = 0.
          scale = 2.
          var_epsilon = 1e-5

          bn = tf.nn.batch_normalization(t_input0, mean, var, offset, scale, var_epsilon)
          self.assertAllEqual(bn, (scale/math.sqrt(var)) * input0)

class MIopenBn1D(tf.test.TestCase):

  def _test(self):
      with self.test_session():
        sizes = [15,64,512,8096,32565,65535]

        for size in sizes:
          input0 = tf.random.uniform(shape=[size], minval=0, maxval=64, dtype = np.float16)
          t_input0 = tf.convert_to_tensor(input0, np.float16)

          mean = 0.5
          var = 0.1
          offset = 1.5
          scale = 2.
          var_epsilon = 1e-5

          bn = tf.nn.batch_normalization(t_input0, mean, var, offset, scale, var_epsilon)
          #self.assertAllEqual(bn, (scale/math.sqrt(var)) * (input0 - mean) + offset)

class MIopenBn4D(tf.test.TestCase):

  def _test(self):
      with self.test_session():
        sizes = [(1,8,8,3)]

        for size in sizes:
          input0 = tf.random.uniform(shape=[192], minval=0, maxval=64, dtype = np.float16)
          input0 = tf.reshape(input0,size)
          t_input0 = tf.convert_to_tensor(input0, np.float16)

          offset = 0.
          scale = 1.
          var_epsilon = 1e-5
          mean = tf.reduce_mean(t_input0,axis=[0,1,2])
          var = tf.math.reduce_std(t_input0,axis=[0,1,2])

          #mean = 0.0
          #var = 1.0

          mean = tf.zeros(mean.shape,dtype = tf.dtypes.float16)
          var = tf.ones(var.shape,dtype = tf.dtypes.float16)

          bn = tf_fim_ops.miopen_bn(t_input0, mean, var, offset, scale, var_epsilon)
          #bn = tf.nn.batch_normalization(t_input0, mean, var, offset, scale, var_epsilon)
          self.assertAllEqual(bn, input0)

if __name__ == '__main__':
    tf.test.main()
