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

class PimMulTestRandom(tf.test.TestCase):
    def test_4dim_4dim(self):
        batch_size = [1, 4, 8]
        channel = [1, 12]
        width = [128, 256, 384]
        height = [768, 1024]

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
                                50,
                                maxval=50,
                                dtype=np.float16)
                            input1 = tf.random.uniform(
                                shape=[
                                    b,
                                    c,
                                    w,
                                    h],
                                minval=-
                                50,
                                maxval=50,
                                dtype=np.float16)
                            mul = tf.constant([1], dtype=np.int32)
                            tf_pim_ops.pim_init()
                            result_custom = tf_pim_ops.pim_eltwise(input0, input1, mul)
                            tf_pim_ops.pim_deinit()
                            result_math_multiply = tf.math.multiply(
                                input0, input1)
                            
                            try:
                                self.assertAllEqual(
                                    result_custom, result_math_multiply)
                            except Exception as ex:
                                failed_cases.append([b, c, w, h])
                                success = False
                            
                            trace_file_name = "trace_file_" + str(b)+ "x"+str(c) + "x" + str(w) + "x" + str(h)
                            golden_file_name = "golden_file_" + str(b)+ "x"+str(c) + "x" + str(w) + "x" + str(h)
                            shutil.copyfile("/tmp/pim_rtl/mem_trace_debug.txt",  "rtl_tv/" + trace_file_name)
                            shutil.copyfile("/tmp/pim_rtl/rtl_output_golden.txt",  "rtl_tv/" + golden_file_name)

        if not success:
            print("Test cases failed!: " + str(failed_cases))
        self.assertEqual(success, True)

if __name__ == '__main__':
    tf.test.main()
    
