from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import logging

from tensorflow.python.platform import test
try:
  from tf_fim_ops.python.ops.fim_gemv_ops import fim_gemv
except ImportError:
  from fim_gemv_ops import fim_gemv

tf.debugging.set_log_device_placement(True)
testFilesPath = '../test_vectors/'

class GemvTest(tf.test.TestCase):
  def testGemvConstant(self):
    in_size = 256
    out_size =  4096
    an = np.ones(shape=(1,in_size))
    bn = np.ones(shape=(in_size,out_size))
    a =  tf.constant(an,dtype = np.float16)
    b =  tf.constant(bn,dtype = np.float16)
    with self.test_session():
        #result = tf.linalg.matmul(a,b)
        result = fim_gemv(a,b,tf.constant([1]))
        logging.info('Result %s',result)
        logging.info('Result %s',result.shape)
        logging.info('Result %s %s %s',result[0],result[254],result[257])
        self.assertAllEqual(result, [256.]*out_size)


  def testGemvGolden(self):
    in_size = 256
    out_size =  4096
    input0  = np.fromfile(testFilesPath + "load/gemv/input_256x1.dat",dtype = np.float16)
    input1  = np.fromfile(testFilesPath + "load/gemv/weight_256x4096.dat",dtype = np.float16)
    output0 = np.fromfile(testFilesPath + "load/gemv/output_4096x1.dat",dtype = np.float16)

    t_input0   = tf.convert_to_tensor(input0, np.float16)
    t_input1   = tf.convert_to_tensor(input1, np.float16)
    t_output0  = tf.convert_to_tensor(output0, np.float16)

    a = tf.reshape(t_input0, [1, in_size])
    w = tf.reshape(t_input1, [in_size, out_size])
    o = tf.reshape(t_output0,[out_size])

    with self.test_session():
        result = fim_gemv(a,w,tf.constant([1]))
        logging.info('Result %s',result)
        logging.info('Result %s',result.shape)
        logging.info('Result1 %s',result[0])
        self.assertAllEqual(result,o)


  def testGemvGoldenWeightReordered(self):
    in_size = 256
    out_size =  4096
    input0  = np.fromfile(testFilesPath + "load/gemv/input_256x1.dat",dtype = np.float16)
    input1  = np.fromfile(testFilesPath + "load/gemv/reordered_weight_256x4096.dat",dtype = np.float16)
    output0 = np.fromfile(testFilesPath + "load/gemv/output_4096x1.dat",dtype = np.float16)

    t_input0   = tf.convert_to_tensor(input0, np.float16)
    t_input1   = tf.convert_to_tensor(input1, np.float16)
    t_output0  = tf.convert_to_tensor(output0, np.float16)

    a = tf.reshape(t_input0, [1, in_size])
    w = tf.reshape(t_input1, [in_size, out_size])
    o = tf.reshape(t_output0,[out_size])

    with self.test_session():
        result = fim_gemv(a,w,tf.constant([0]))
        logging.info('Result %s',result)
        logging.info('Result %s',result.shape)
        logging.info('Result1 %s',result[0])
        self.assertAllEqual(result,o)


  def _testGemvSmall(self):
    an = np.ones(shape=(1,64))
    bn = np.ones(shape=(64,32))

    an = an * 1
    bn = bn * 1
    a =  tf.constant(an,dtype = np.float16)
    b =  tf.constant(bn,dtype = np.float16)
    with self.test_session():
      result = fim_gemv(a,b)
      logging.info('Result %s',result)
      logging.info('Result %s',result.shape)
      self.assertAllEqual(result, [64.]*32)

if __name__ == "__main__":
  tf.test.main()
