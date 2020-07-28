from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import tensorflow as tf
import numpy as np
import tf_fim_ops

tf.keras.backend.set_floatx('float16')
tf.debugging.set_log_device_placement(True)
SEED = 1234

class LstmTest(tf.test.TestCase):
    def testLstmLatency(self):

        # configs
        miopen_num_layers = num_layers = 5
        bidirectional = True
        hidden_size = 800
        n_batch = 1
        n_seq = 50
        n_input = 2592
        n_iters = 10
        ws_len = 9600000 * 2 #hack to avoid malloc everytime in custom_op


        bi = 1
        if bidirectional:
            miopen_num_layers = num_layers * 2
            bi = 2

        inputs = tf.random.uniform(
            shape=(2,
                n_batch,
                n_seq,
                n_input),
            dtype=tf.float16)
        hidden_states = tf.random.uniform(
            shape=(2,
                miopen_num_layers,
                inputs.shape[1],
                hidden_size),
            dtype=tf.float16)
        cell_states = tf.random.uniform(
            shape=(2,
                miopen_num_layers,
                inputs.shape[1],
                hidden_size),
            dtype=tf.float16)

        # Calculate weight dim's referring Miopen driver's /src/rnn.cpp
        weight_x = inputs.shape[3] + \
            ((num_layers  - 1) * (bi + 1) + 1) * hidden_size
        weight_y = bi * hidden_size * 4  # nHiddenTensorsPerLayer;
        weights = tf.random.uniform(
            shape=(2,
                weight_x,
                weight_y),
            dtype=tf.float16)

        with self.test_session():

            for i in range(n_iters):
              start = datetime.datetime.now()
              result, hidden_out, cell_out, ws = tf_fim_ops.fim_lstm(
                  inputs, weights, hidden_states, cell_states, tf.constant([bi]),tf.constant([ws_len]))
              end = datetime.datetime.now()
              duration = end - start
              print('Python Duration:', i , '  ' ,  duration.microseconds/1000)

            #print('Output shape', result.shape)
            #print('Output shape', result.shape)
            #print('Hidden out shape', hidden_out.shape)
            #print('Cell out shape', cell_out.shape)
            #self.assertAllEqual(result, o.reshape(1, out_size))


    def testkeras(self):

        miopen_num_layers= num_layers = 5
        bidirectional = True
        hidden_size = 800

        n_batch = 1
        n_seq = 50
        n_input = 2592
        n_iters = 5
        ws_len = 9600000 * 4 #hack to avoid malloc everytime in custom_op

        cell_val = 0.0
        hid_val = 0.0
        weight_val = 0.001

        bi = 1
        if bidirectional:
            miopen_num_layers = num_layers * 2
            bi = 2

        inputs = tf.random.uniform(
            shape=(2,
                n_batch,
                n_seq,
                n_input),
            dtype=tf.float16)
        hidden_states = tf.constant(
            hid_val,
            shape=(2,
                miopen_num_layers,
                inputs.shape[1],
                hidden_size),
            dtype=tf.float16)
        cell_states = tf.constant(
            cell_val,
            shape=(2,
                miopen_num_layers,
                inputs.shape[1],
                hidden_size),
            dtype=tf.float16)

        weight_x = inputs.shape[3] + \
            ((num_layers - 1) * (bi + 1) + 1) * hidden_size
        weight_y = bi * hidden_size * 4  # nHiddenTensorsPerLayer;
        weights = tf.constant(
            weight_val,
            shape=(2,
                weight_x,
                weight_y),
            dtype=tf.float16)


        #print(weights.shape)
        #print(inputs.shape)
        #print(cell_states.shape)
        #print(hidden_states.shape)
        for i in range(n_iters):
          start = datetime.datetime.now()
          result, hidden_out, cell_out, ws = tf_fim_ops.fim_lstm(
                 inputs, weights, hidden_states, cell_states, tf.constant([bi]),tf.constant([ws_len]))
          end = datetime.datetime.now()
          duration = end - start
          print('Python Duration:' , '  ' ,  duration.microseconds/1000)
        #print('Fim.lstm value:',result)
        lstm = tf.keras.Sequential()
        lstm.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size,
                            kernel_initializer=tf.keras.initializers.Constant(weight_val),
                            recurrent_initializer=tf.keras.initializers.Constant(weight_val),
                            return_sequences=True,
                            dtype='float16',
                            trainable=False)))
        for i in range(num_layers - 1):
            lstm.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size,
                            kernel_initializer=tf.keras.initializers.Constant(weight_val),
                            recurrent_initializer=tf.keras.initializers.Constant(weight_val),
                            return_sequences=True,
                            dtype='float16',
                            trainable=False)))
        for i in range(5):

          start = datetime.datetime.now()
          whole_seq_out = lstm(inputs[0],training=False)
          end = datetime.datetime.now()
          duration = end - start
          print('Python Duration:',  '  ' ,  duration.microseconds/1000)
          #print('keras lstm value:',whole_seq_out)
          self.assertAllClose(whole_seq_out[0], result[0], atol=1e-1)

    #currently disabled , useful to check with miopen golden
    def _testmiopen(self):

        miopen_num_layers= num_layers = 5
        bidirectional = True
        hidden_size = 800

        n_batch = 1
        n_seq = 50
        n_input = 2592
        n_iters = 3
        ws_len = 96000000 * 2 #hack to avoid malloc everytime in custom_op

        bi = 1
        if bidirectional:
            miopen_num_layers = num_layers * 2
            bi = 2

       #NOTE: We append 2 to all shapes since miopen needs 2x memory

        inputs = np.fromfile('dump_in.bin',dtype=np.float16)
        inputs = np.reshape(inputs,(2,n_batch,n_seq,n_input))
        inputs = tf.convert_to_tensor(inputs)

        weight_x = inputs.shape[3] + \
            ((num_layers - 1) * (bi + 1) + 1) * hidden_size
        weight_y = bi * hidden_size * 4  # nHiddenTensorsPerLayer;
        weights = np.fromfile('dump_wei.bin',dtype=np.float16)
        weights = np.reshape(weights,(2,weight_x,weight_y))

        hidden_states = np.fromfile('dump_hx.bin',dtype=np.float16)
        hidden_states = np.reshape(hidden_states,(2,miopen_num_layers,inputs.shape[1],hidden_size))
        hidden_states = tf.convert_to_tensor(hidden_states)

        cell_states = np.fromfile('dump_cx.bin',dtype=np.float16)
        cell_states = np.reshape(cell_states,(2,miopen_num_layers,inputs.shape[1],hidden_size))
        cell_states = tf.convert_to_tensor(cell_states)

        outputs = np.fromfile('dump_fwd_out_gpu.bin',dtype=np.float16)
        outputs = np.reshape(outputs,(2,n_seq,2*hidden_size))
        golden = tf.convert_to_tensor(outputs)

        with self.test_session():

          for i in range(n_iters):
            start = datetime.datetime.now()
            result, hidden_out, cell_out, ws = tf_fim_ops.fim_lstm(
               inputs, weights, hidden_states, cell_states, tf.constant([bi]),tf.constant([ws_len]))
            end = datetime.datetime.now()
            duration = end - start
            print('Python Duration:', i , '  ' ,  duration.microseconds/1000)

        #NOTE: check only 0th index outputs array , rest is not filled.
        self.assertAllClose(result[0], golden[0], atol=1e-2)


if __name__ == "__main__":
    tf_fim_ops.fim_init()
    tf.test.main()
    tf_fim_ops.fim_deinit()
