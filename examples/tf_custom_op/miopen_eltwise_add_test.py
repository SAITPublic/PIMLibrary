from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.python.platform import test
try:
  from tf_fim_ops.python.ops.miopen_eltwise_ops import miopen_elt
except ImportError:
  from miopen_eltwise_ops import miopen_elt

tf.debugging.set_log_device_placement(True)
testFilesPath = '../test_vectors/'


class MIopenAddTestConstant(tf.test.TestCase):

  def test(self):
      with self.test_session():
        input0  =  tf.constant([1.,2.,3.,4.,5.],dtype = np.float16)
        t_input0   = tf.convert_to_tensor(input0, np.float16)

        input1  =  tf.constant([50.,5.,5.,5.,5.],dtype = np.float16)
        t_input1   = tf.convert_to_tensor(input1, np.float16)

        add = tf.constant([0],dtype = np.int32)
        result = miopen_elt(t_input0,t_input1,add)
        self.assertAllEqual(result, [51.,7.,8.,9.,10.])

class MIopenAddTestRandom(tf.test.TestCase):

  def test(self):
      with self.test_session():
        input0 = tf.random.uniform(shape=[65536], minval=0, maxval=64, dtype = np.float16)
        t_input0 = tf.convert_to_tensor(input0, np.float16)

        input1 = tf.random.uniform(shape=[65536], minval=0, maxval=64, dtype = np.float16)
        t_input1   = tf.convert_to_tensor(input1, np.float16)

        add = tf.constant([0],dtype = np.int32)
        result_custom = miopen_elt(t_input0, t_input1,add)
        result_math_add = tf.math.add(t_input0, t_input1)

        self.assertAllEqual(result_custom, result_math_add)

class MIopenAddTestFile(tf.test.TestCase):

  def test(self):
      with self.test_session():

        input0  =  np.fromfile("../test_vectors/load/elt_add/input0_128KB.dat", dtype = np.float16)
        t_input0   = tf.convert_to_tensor(input0, np.float16)

        input1  =  np.fromfile("../test_vectors/load/elt_add/input1_128KB.dat", dtype = np.float16)
        t_input1   = tf.convert_to_tensor(input1, np.float16)

        add = tf.constant([0],dtype = np.int32)
        result = miopen_elt(t_input0,t_input1,add)
        golden  =  np.fromfile("../test_vectors/load/elt_add/output_128KB.dat", dtype = np.float16)
        self.assertAllEqual(result, golden)

if __name__ == '__main__':
  tf.test.main()


