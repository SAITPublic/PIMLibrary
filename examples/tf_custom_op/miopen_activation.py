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

class MIopenActTestFile(tf.test.TestCase):

  def test(self):
      with self.test_session():

        input0  =  np.fromfile("../test_vectors/load/relu/input_256KB.dat", dtype = np.float16)
        t_input0   = tf.convert_to_tensor(input0, np.float16)

        result = miopen_act(t_input0)
        golden  =  np.fromfile("../test_vectors/load/relu/output_256KB.dat", dtype = np.float16)
        self.assertAllEqual(result, golden)

if __name__ == '__main__':
  tf.test.main()


