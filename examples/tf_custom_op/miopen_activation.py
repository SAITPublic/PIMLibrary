from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.python.platform import test
try:
  from tf_fim_ops.python.ops.miopen_activation_ops import miopen_act
except ImportError:
  from miopen_activation_ops import miopen_act

tf.debugging.set_log_device_placement(True)
testFilesPath = '../test_vectors/'


class MIopenActTestRandom(tf.test.TestCase):
  def test(self):
    batch_size = [1, 10]
    channel = [32, 128, 384]
    width = [3, 8, 73]
    height = [3, 8, 73]
    failed_cases = []
    success = True

    for b in batch_size:
      for c in channel:
        for w in width:
          for h in height:
            with self.test_session():
              input0 = tf.random.uniform(shape=[b, c, w, h], minval=-500, maxval=500, dtype=np.float16)
              result_custom = miopen_act(input0)
              result_tf_relu = tf.nn.relu(input0)
              try:
                self.assertAllEqual(result_custom, result_tf_relu)
              except Exception as ex:
                failed_cases.append([b, c, w, h])
                success = False
    print("Test cases failed!: " + str(failed_cases))
    self.assertEqual(success, True)


class MIopenActTestFile(tf.test.TestCase):
  def test(self):
      with self.test_session():

        input0  =  np.fromfile("../test_vectors/load/relu/input_256KB.dat", dtype = np.float16)
        result = miopen_act(input0)
        golden  =  np.fromfile("../test_vectors/load/relu/output_256KB.dat", dtype = np.float16)
        self.assertAllEqual(result, golden)


if __name__ == '__main__':
  tf.test.main()


