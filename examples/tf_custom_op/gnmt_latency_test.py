import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import tf_fim_ops

tf.keras.backend.set_floatx('float16')
SEED = 1234
BATCH_SIZE = 1
MAX_LENGTH = 50
HIDDEN_UNITS = 1024
NUM_ITER = 5
VOCAB_SIZE = 32000

def profile_gnmt(batch_size=1):
    input_seq = tf.random.uniform(shape=(batch_size, MAX_LENGTH), dtype=tf.float16)
    print('Input Shape: (batch_size, timestep, units){}'.format(input_seq.shape))

    embedded_sequence = tf.keras.layers.Embedding(VOCAB_SIZE, HIDDEN_UNITS, input_length=MAX_LENGTH)(input_seq)
    print('Embedded Shape: (batch_size, timestep, units){}'.format(embedded_sequence.shape))


    eval_time = []
    for i in range(NUM_ITER):
        #Encoder
        print('Iteration number : (iteration) {}'.format(i))
        it_start = time.time()
        start = time.time()
        output = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(HIDDEN_UNITS,
                               kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                               recurrent_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                               return_sequences=True,
                               dtype='float16',
                               trainable=False))(embedded_sequence)
        output = tf.keras.layers.LSTM(HIDDEN_UNITS,
                               kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                               recurrent_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                               return_sequences=True,
                               dtype='float16',
                               trainable=False)(output)
        output  = tf.keras.layers.LSTM(HIDDEN_UNITS,
                               kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                               recurrent_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                               return_sequences=True,
                               dtype='float16',
                               trainable=False)(output)
        output, memory_state, carry_state = tf.keras.layers.LSTM(HIDDEN_UNITS,
                               kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                               recurrent_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                               return_sequences=True,
                               return_state=True,
                               dtype='float16',
                               trainable=False)(output)

        print('Time taken for Encoder {} sec\n'.format(time.time() - start))
        print('encoder dimensions: (batch_size, timestep, units){}'.format(output.shape))
        print('hidden dimensions: (batch_size, units){}'.format(memory_state.shape))

        # Attention Layer
        start = time.time()
        output = tf.keras.layers.AdditiveAttention(1)([output, memory_state])
        print('Time taken for Attention {} sec\n'.format(time.time() - start))

        # Decoder Layer
        start = time.time()
        output = tf.keras.layers.LSTM(HIDDEN_UNITS,
                               kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                               recurrent_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                               return_sequences=True,
                               dtype='float16',
                               trainable=False)(output)
        output = tf.keras.layers.LSTM(HIDDEN_UNITS,
                               kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                               recurrent_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                               return_sequences=True,
                               dtype='float16',
                               trainable=False)(output)
        output = tf.keras.layers.LSTM(HIDDEN_UNITS,
                               kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                               recurrent_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                               return_sequences=True,
                               dtype='float16',
                               trainable=False)(output)
        output = tf.keras.layers.LSTM(HIDDEN_UNITS,
                               kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                               recurrent_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                               return_sequences=True,
                               dtype='float16',
                               trainable=False)(output)
        print('Time taken for Decoder {} sec\n'.format(time.time() - start))
        eval_time.append(time.time() - it_start)

    print('Time taken for iteration {} sec\n'.format(eval_time))

tf_fim_ops.fim_init()
profile_gnmt(BATCH_SIZE)
tf_fim_ops.fim_deinit()
