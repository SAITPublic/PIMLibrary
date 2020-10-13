import tensorflow as tf
from tabulate import tabulate

import time
import tf_fim_ops
import timeit

SEED = 1234

# GNMT model configuration
HIDDEN_SIZE = 1024
MAX_SEQ_LENGTH = 100
EMBEDDING_DIM = 1024
VOCAB_SIZE = 32000
BATCH_SIZE = 1

# Number of evaluation iterations
NUM_ITERATIONS = 10

tf.keras.backend.set_floatx('float16')

# Performance table for different layers
eval_time = []
profile = True



# Encoder class GNMT model
class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, initializer):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    self.lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(enc_units,
                           kernel_initializer=initializer,
                           recurrent_initializer=initializer,
                           return_sequences=True,
                           dtype='float16',
                           trainable=False))
    self.lstm2 = tf.keras.layers.LSTM(enc_units,
                           kernel_initializer=initializer,
                           recurrent_initializer=initializer,
                           return_sequences=True,
                           dtype='float16',
                           trainable=False)
    self.lstm3 = tf.keras.layers.LSTM(enc_units,
                           kernel_initializer=initializer,
                           recurrent_initializer=initializer,
                           return_sequences=True,
                           dtype='float16',
                           trainable=False)
    self.lstm4 = tf.keras.layers.LSTM(enc_units,
                           kernel_initializer=initializer,
                           recurrent_initializer=initializer,
                           return_sequences=True,
                           return_state=True,
                           dtype='float16',
                           trainable=False)

  def lstm_encoder(self, x):
    output = self.lstm1(x)
    output = self.lstm2(output)
    output = self.lstm3(output)
    output, h_state, c_state = self.lstm4(output)

    return output, h_state, c_state

  def call(self, input_seq, hidden):
    x = self.embedding(input_seq)

    if profile == True:
        eval_time.append(["Encoder Embedding", (timeit.timeit(lambda : self.embedding(input_seq), number = NUM_ITERATIONS)) * 1000])
        print('encoder embed output dimensions(batch, timestep, units): {}'.format(x.shape))

    output, h_state, c_state = self.lstm_encoder(x)

    if profile == True:
        eval_time.append(["Encoder LSTM", (timeit.timeit(lambda : self.lstm_encoder(x), number = NUM_ITERATIONS)) * 1000])

    return output, h_state, c_state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

# Bahdanau Style Additive attention implementation
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

# Decoder implementation for GNMT model
class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, initializer):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.lstm1 = tf.keras.layers.LSTM(dec_units,
                               kernel_initializer=initializer,
                               recurrent_initializer=initializer,
                               return_sequences=True,
                               dtype='float16',
                               trainable=False)
    self.lstm2 = tf.keras.layers.LSTM(dec_units,
                               kernel_initializer=initializer,
                               recurrent_initializer=initializer,
                               return_sequences=True,
                               dtype='float16',
                               trainable=False)
    self.lstm3 = tf.keras.layers.LSTM(dec_units,
                               kernel_initializer=initializer,
                               recurrent_initializer=initializer,
                               return_sequences=True,
                               dtype='float16',
                               trainable=False)
    self.lstm4 = tf.keras.layers.LSTM(dec_units,
                               kernel_initializer=initializer,
                               recurrent_initializer=initializer,
                               return_sequences=True,
                               return_state=True,
                               dtype='float16',
                               trainable=False)
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def lstm_decoder(self, context_vector, x):
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    output = self.lstm1(x)
    x = tf.concat([tf.expand_dims(context_vector, 1), output], axis=-1)
    output = self.lstm2(x)
    x = tf.concat([tf.expand_dims(context_vector, 1), output], axis=-1)
    output = self.lstm3(x)
    x = tf.concat([tf.expand_dims(context_vector, 0), output], axis=-1)
    
    return self.lstm4(x)

  def call(self, x, hidden, enc_output, dec_dict):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    if profile == True:
        dec_dict["Attention"] += (timeit.timeit(lambda: self.attention(hidden, enc_output), number = NUM_ITERATIONS) * 1000)
    # print('Context vector dimensions: {}'.format(context_vector.shape))
    # print('Attention weights dimensions: {}'.format(attention_weights.shape))

    start = time.time()
    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)
    
    if profile == True:
        dec_dict["Decoder Embedding"] += (timeit.timeit(lambda: self.embedding(x), number = NUM_ITERATIONS) * 1000)
    # print('Decoder embedding dimensions: {}'.format(x.shape))

    output , h_state, c_state = self.lstm_decoder(context_vector, x)

    if profile == True:
        dec_dict["Decoder LSTM"] += (timeit.timeit(lambda: self.lstm_decoder(context_vector, x), number = NUM_ITERATIONS) * 1000)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    # passing the concatenated vector to the LSTM
    # print('Decoder LSTM dimensions: (output),(hidden_state),(carry_state){}'.format(output.shape))
    # output shape == (batch_size * 1, hidden_size)
    reshaped_output = tf.reshape(output, (-1, output.shape[2]))

    if profile == True:
        dec_dict["Reshape"] += (timeit.timeit(lambda: tf.reshape(output, (-1, output.shape[2])), number=NUM_ITERATIONS) * 1000)
    # print('Reshape dimensions: {}'.format(output.shape))
    # output shape == (batch_size, vocab)

    x = self.fc(reshaped_output)

    if profile == True:
        dec_dict["Dense"] += (timeit.timeit(lambda: self.fc(output),number=NUM_ITERATIONS) * 1000)
    # print('Dense output dimensions: {}'.format(x.shape))

    return x, h_state, attention_weights

def create_gnmt_model(vocab_size, embed_dim, hidden, max_len, batch_size, initializer):
    encoder = Encoder(vocab_size, embed_dim, hidden, batch_size, initializer)
    decoder = Decoder(vocab_size, embed_dim, hidden, batch_size, initializer)

    return encoder, decoder

def evaluate(sentence, encoder, decoder, max_length_targ=MAX_SEQ_LENGTH):

    inputs = tf.convert_to_tensor(sentence)

    h_state = [tf.zeros((1, HIDDEN_SIZE))]
    c_state = [tf.zeros((1, HIDDEN_SIZE))]

    enc_out, enc_hidden, enc_carry = encoder(inputs, [h_state, c_state])

    if profile == True :
        print('Input dimensions: (batch_size, timestep){}'.format(inputs.shape))
        print('encoder output dimensions: {}'.format(enc_out.shape))
        print('encoder final hidden state dimensions: {}'.format(enc_hidden.shape))
        print('encoder final carry state dimensions: {}'.format(enc_carry.shape))

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([0], 0)

    # for profiling of decoder
    dec_dict = {"Attention": 0, "Decoder Embedding": 0, "Decoder LSTM": 0, "Reshape": 0, "Dense": 0}

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out,
                                                         dec_dict)

        predicted_id = tf.argmax(predictions[0]).numpy()

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    for key, val in dec_dict.items():
        eval_time.append([key,val])

    return predictions

def gnmt_model_run():
    tf_fim_ops.fim_init()

    initializer = tf.keras.initializers.RandomNormal(seed=SEED)
    input_seq   = tf.random.uniform(shape=(BATCH_SIZE, MAX_SEQ_LENGTH), dtype=tf.float16)

    encoder, decoder = create_gnmt_model(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, MAX_SEQ_LENGTH, BATCH_SIZE, initializer)

    # model and gpu initialization
    eval_time.clear()
    predictions = evaluate(input_seq, encoder, decoder)

    encoder.summary()
    decoder.summary()

    eval_time.clear()
    predictions = evaluate(input_seq, encoder, decoder)

    # for disabling internal profiling calls.
    #global profile
    #profile = False
    #eval_time.append(["End to End", (timeit.timeit(lambda : evaluate(input_seq, encoder, decoder), number = NUM_ITERATIONS) * 1000)])

    for i in range(len(eval_time)):
        eval_time[i][1] /= NUM_ITERATIONS

    print(tabulate(eval_time, headers=["Index", "Layer", "Time(ms)"], showindex="always", tablefmt='github'))

    tf_fim_ops.fim_deinit()

if __name__ == '__main__':
    gnmt_model_run()
