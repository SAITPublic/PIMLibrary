from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import shutil
import tensorflow as tf
import numpy as np
import tf_pim_ops

tf.debugging.set_log_device_placement(True)
np.set_printoptions(threshold=sys.maxsize)

class PimActTestRandom(tf.test.TestCase):
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
                            input0 = tf.random.uniform(
                                shape=[
                                    b,
                                    c,
                                    w,
                                    h],
                                minval=-
                                500,
                                maxval=500,
                                dtype=np.float16)

                            tf_pim_ops.pim_init()
                            result_custom = tf_pim_ops.pim_act(input0)
                            result_tf_relu = tf.nn.relu(input0)
                            tf_pim_ops.pim_deinit()

                            self.assertAllEqual(
                                    result_custom, result_tf_relu)

                            trace_file_name = "trace_file_" + str(b)+ "x"+str(c) + "x" + str(w) + "x" + str(h)
                            golden_file_name = "golden_file_" + str(b)+ "x"+str(c) + "x" + str(w) + "x" + str(h)
                            shutil.copyfile("/tmp/pim_rtl/mem_trace_debug.txt",  "rtl_tv/" + trace_file_name)
                            shutil.copyfile("/tmp/pim_rtl/rtl_output_golden.txt",  "rtl_tv/" + golden_file_name)

if __name__ == '__main__':

    tf.test.main()

