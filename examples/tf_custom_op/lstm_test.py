import tensorflow as tf
import numpy as np
import math
import os
from os import path
from tensorflow.python.platform import test
tf.debugging.set_log_device_placement(True)
tf.keras.backend.set_floatx('float16')

NUM_UNITS = 512
BATCH = 2
SEQ_LEN= 1
FEATURES = 1024
SEED = 1234

folder = './lstm/'

def gen_constant(dump):

    inputs = np.ones([BATCH, SEQ_LEN, FEATURES]).astype(np.float16)
    lstm = tf.keras.layers.LSTM(NUM_UNITS,
                            kernel_initializer=tf.keras.initializers.Constant(0.1),
                            recurrent_initializer='zeros',
                            return_sequences=True,
                            return_state=True,
                            dtype='float16',
                            trainable=False)

    whole_seq_out, final_mem_state, c_state = lstm(inputs,training=False)
    if dump:
        whole_seq_out.numpy().tofile(folder+'lstm_constant_out.bin')
        inputs.tofile(folder+'lstm_constant_in.bin')

def gen_random(dump):

    inputs = np.ones([BATCH, SEQ_LEN, FEATURES]).astype(np.float16)
    lstm = tf.keras.layers.LSTM(NUM_UNITS,
                            kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                            recurrent_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                            return_sequences=True,
                            return_state=True,
                            dtype='float16',
                            trainable=False)

    whole_seq_out, final_mem_state, c_state = lstm(inputs,training=False)
    if dump:
        whole_seq_out.numpy().tofile(folder+'lstm_random_out.bin')
        inputs.tofile(folder+'lstm_random_in.bin')

class LstmConstant(tf.test.TestCase):

  def test(self):
    with self.test_session():

      inputs = np.ones([BATCH, SEQ_LEN, FEATURES]).astype(np.float16)
      lstm = tf.keras.layers.LSTM(NUM_UNITS,
                            kernel_initializer=tf.keras.initializers.Constant(0.1),
                            recurrent_initializer='zeros',
                            return_sequences=True,
                            return_state=True,
                            dtype='float16',
                            trainable=False)

      whole_seq_out, final_mem_state, c_state = lstm(inputs,training=False)
      golden =np.fromfile(folder+'lstm_constant_out.bin',dtype=np.float16)
      golden  = np.reshape(golden,(BATCH,SEQ_LEN,NUM_UNITS))
      result =  np.array_equal(whole_seq_out,golden)

class LstmRandom(tf.test.TestCase):

  def test(self):
    with self.test_session():

      inputs = np.ones([BATCH, SEQ_LEN, FEATURES]).astype(np.float16)
      lstm = tf.keras.layers.LSTM(NUM_UNITS,
                            kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                            recurrent_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                            return_sequences=True,
                            return_state=True,
                            dtype='float16',
                            trainable=False)

      whole_seq_out, final_mem_state, c_state = lstm(inputs,training=False)
      golden = np.fromfile(folder+'lstm_random_out.bin',dtype=np.float16)
      golden = np.reshape(golden,(BATCH,SEQ_LEN,NUM_UNITS))
      result = np.array_equal(whole_seq_out,golden)


if __name__ == '__main__':

  if not path.exists(folder):
    os.mkdir(folder)
  if path.isfile(folder+'lstm_constant_out.bin') == False:
    gen_constant(True)
  if path.isfile(folder+'lstm_random_out.bin') == False:
    gen_random(True)

  tf.test.main()
