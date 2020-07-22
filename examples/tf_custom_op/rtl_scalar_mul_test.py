from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import numpy as np
import tf_fim_ops

tf.debugging.set_log_device_placement(True)
testFilesPath = 'test_vectors/'
np.set_printoptions(threshold=sys.maxsize)

class FimMulTestRandom(tf.test.TestCase):
    def test_4dim_scalar(self):
        batch_size = [1, 8]
        channel = [1, 12]
        width = [128, 384]
        height = [128, 384]
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
                            input1 = tf.constant([7], dtype=np.float16)

                            mul = tf.constant([1], dtype=np.int32)
                            tf_fim_ops.fim_init()
                            result_custom = tf_fim_ops.fim_eltwise(input0, input1, mul)
                            result_math_multiply = tf.math.multiply(
                                input0, input1)
                            tf_fim_ops.fim_deinit()
                            try:
                                self.assertAllEqual(
                                    result_custom, result_math_multiply)
                            except Exception as ex:
                                failed_cases.append([b, c, w, h])
                                success = False

                            trace_file_name = "trace_file_" + str(b)+ "x"+str(c) + "x" + str(w) + "x" + str(h)
                            golden_file_name = "golden_file_" + str(b)+ "x"+str(c) + "x" + str(w) + "x" + str(h)
                            os.rename("/home/user/fim-workspace/ssh/FIMLibrary/test_vectors/rtl/mem_trace_debug.txt" ,
                                    "/home/user/fim-workspace/ssh/FIMLibrary/test_vectors/rtl/" + trace_file_name)
                            os.rename("/home/user/fim-workspace/ssh/FIMLibrary/test_vectors/rtl/rtl_output_golden.txt",
                                    "/home/user/fim-workspace/ssh/FIMLibrary/test_vectors/rtl/" + golden_file_name)
        if not success:
            print("Test cases failed!: " + str(failed_cases))
        self.assertEqual(success, True)

if __name__ == '__main__':
    tf.test.main()
    
