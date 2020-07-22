from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import numpy as np
import tf_fim_ops

tf.debugging.set_log_device_placement(True)
testFilesPath = '../test_vectors/'
np.set_printoptions(threshold=sys.maxsize)

class GemvTest(tf.test.TestCase):

    def testGemvCoverage(self):
        with self.test_session():
            minv = -1.0
            maxv = 1.0 
            batches = [1, 4]
            sizes = [
                (128, 768),
                (256, 768),
                (384, 768),
                (128, 1024),
                (256, 1024),
                (384, 1024),
                (800, 3200)
            ]
            success = True
            failed_cases = []
            for batch in batches:
              for size in sizes:
                a = tf.random.uniform(
                    shape=(
                        batch,
                        size[0]),
                    minval=minv,
                    maxval=maxv,
                    dtype=tf.dtypes.float16)
                b = tf.random.uniform(
                    shape=(
                        size[0],
                        size[1]),
                    minval=minv,
                    maxval=maxv,
                    dtype=tf.dtypes.float16)
                tf_fim_ops.fim_init()
                golden = tf.linalg.matmul(a, b)
                result = tf_fim_ops.fim_gemv(a, b, tf.constant([1]))
                tf_fim_ops.fim_deinit()
                
                trace_file_name = "trace_file_" + str(batch)+ "x"+str(size[0]) + "x" + str(size[1])
                golden_file_name = "golden_file_" + str(batch)+ "x"+str(size[0]) + "x" + str(size[1])
                os.rename("/home/user/fim-workspace/ssh/FIMLibrary/test_vectors/rtl/mem_trace_debug.txt" ,
                    "/home/user/fim-workspace/ssh/FIMLibrary/test_vectors/rtl/" + trace_file_name)
                os.rename("/home/user/fim-workspace/ssh/FIMLibrary/test_vectors/rtl/rtl_output_golden.txt",
                    "/home/user/fim-workspace/ssh/FIMLibrary/test_vectors/rtl/" + golden_file_name)

                try:
                    self.assertAllClose(result, golden, rtol=5e-1)
                except Exception as ex:
                    failed_cases.append([batch, size])
                    success = False

            if not success:
                print("Test cases failed!: " + str(failed_cases))
            self.assertEqual(success, True)

if __name__ == "__main__":
    tf.test.main()


