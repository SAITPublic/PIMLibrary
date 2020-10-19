import tensorflow as tf
from tabulate import tabulate

import time
import tf_fim_ops
import timeit
import argparse

tf.keras.backend.set_floatx('float16')

SEED = 1234

# GNMT model configuration
HIDDEN_SIZE = 1024
EMBEDDING_DIM = 1024
VOCAB_SIZE = 32000
# Performance table for different layers
eval_time = []

parser = argparse.ArgumentParser(description='Process GNMT arguments')
parser.add_argument('-b','--batch_size', default=1, help="Input batch size", type=int)
parser.add_argument('-l','--max_seq_length', default=100, help="Maximum sequence length of GNMT input", type=int)
parser.add_argument('-i','--iterations', default=100, help="Number of iterations for profiling", type=int)
parser.add_argument('-p','--profile', action="store_true", help="Enabled/Disable profiling")

args = parser.parse_args()

def DummyExecute():

    # Create some tensors
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)

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

    if args.profile == True:
        eval_time.append(["Encoder Embedding",
                            (timeit.timeit(lambda : self.embedding(input_seq), number = args.iterations)),
                            input_seq.shape, x.shape])
        print('encoder embed output dimensions(batch, timestep, units): {}'.format(x.shape))

    output, h_state, c_state = self.lstm_encoder(x)

    if args.profile == True:
        eval_time.append(["Encoder LSTM",
                            (timeit.timeit(lambda : self.lstm_encoder(x), number = args.iterations)),
                            x.shape, output.shape])

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

    if args.profile == True:
        dec_dict["Attention"]["time"] += (timeit.timeit(lambda: self.attention(hidden, enc_output), number = args.iterations))
        dec_dict["Attention"]["Input"]  = hidden.shape, enc_output.shape
        dec_dict["Attention"]["Output"]  = context_vector.shape
    # print('Context vector dimensions: {}'.format(context_vector.shape))
    # print('Attention weights dimensions: {}'.format(attention_weights.shape))
    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    embed_x = self.embedding(x)
    
    if args.profile == True:
        dec_dict["Decoder Embedding"]["time"] += (timeit.timeit(lambda: self.embedding(x), number = args.iterations))
        dec_dict["Decoder Embedding"]["Input"] = x.shape
        dec_dict["Decoder Embedding"]["Output"] = embed_x.shape
    # print('Decoder embedding dimensions: {}'.format(x.shape))

    output , h_state, c_state = self.lstm_decoder(context_vector, embed_x)

    if args.profile == True:
        dec_dict["Decoder LSTM"]["time"] += (timeit.timeit(lambda: self.lstm_decoder(context_vector, embed_x), number = args.iterations))
        dec_dict["Decoder LSTM"]["Input"] = context_vector.shape, embed_x.shape
        dec_dict["Decoder LSTM"]["Output"] = output.shape

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    # passing the concatenated vector to the LSTM
    # print('Decoder LSTM dimensions: (output),(hidden_state),(carry_state){}'.format(output.shape))
    # output shape == (batch_size * 1, hidden_size)
    reshaped_output = tf.reshape(output, (-1, output.shape[2]))

    if args.profile == True:
        dec_dict["Reshape"]["time"] += (timeit.timeit(lambda: tf.reshape(output, (-1, output.shape[2])), number=args.iterations))
        dec_dict["Reshape"]["Input"] = output.shape
        dec_dict["Reshape"]["Output"] = reshaped_output.shape

    # print('Reshape dimensions: {}'.format(output.shape))
    # output shape == (batch_size, vocab)

    x = self.fc(reshaped_output)

    if args.profile == True:
        dec_dict["Dense"]["time"] += (timeit.timeit(lambda: self.fc(output),number=args.iterations))
        dec_dict["Dense"]["Input"] = reshaped_output.shape
        dec_dict["Dense"]["Output"] = x.shape

    # print('Dense output dimensions: {}'.format(x.shape))

    return x, h_state, attention_weights

def create_gnmt_model(vocab_size, embed_dim, hidden, max_len, batch_size, initializer):
    encoder = Encoder(vocab_size, embed_dim, hidden, batch_size, initializer)
    decoder = Decoder(vocab_size, embed_dim, hidden, batch_size, initializer)

    return encoder, decoder

def evaluate(sentence, encoder, decoder, max_length_targ=args.max_seq_length):

    inputs = tf.convert_to_tensor(sentence)

    h_state = [tf.zeros((1, HIDDEN_SIZE))]
    c_state = [tf.zeros((1, HIDDEN_SIZE))]

    enc_out, enc_hidden, enc_carry = encoder(inputs, [h_state, c_state])

    if args.profile == True :
        print('Input dimensions: (batch_size, timestep){}'.format(inputs.shape))
        print('encoder output dimensions: {}'.format(enc_out.shape))
        print('encoder final hidden state dimensions: {}'.format(enc_hidden.shape))
        print('encoder final carry state dimensions: {}'.format(enc_carry.shape))

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([0], 0)

    if args.profile == True:
        # for profiling of decoder
        dec_dict = {"Attention": { "time": 0, "Input": 0, "Output": 0},
                    "Decoder Embedding": {"time": 0, "Input": 0, "Output": 0},
                    "Decoder LSTM": {"time": 0, "Input": 0, "Output": 0},
                    "Reshape": {"time": 0, "Input": 0, "Output": 0},
                    "Dense": {"time": 0, "Input": 0, "Output": 0},
                    }
    else:
        dec_dict = {}

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out,
                                                         dec_dict)

        predicted_id = tf.argmax(predictions[0]).numpy()

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    if args.profile == True:
        for layer, data in dec_dict.items():
            eval_time.append([layer, data["time"], data["Input"], data["Output"]])

    return predictions

def gnmt_model_run():
    tf_fim_ops.fim_init()

    initializer = tf.keras.initializers.RandomNormal(seed=SEED)
    input_seq   = tf.random.uniform(shape=(args.batch_size, args.max_seq_length), dtype=tf.float16)

    encoder, decoder = create_gnmt_model(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, args.max_seq_length, args.batch_size, initializer)

    if args.profile:
        # model and gpu initialization
        DummyExecute()

    eval_time.clear()
    predictions = evaluate(input_seq, encoder, decoder)

    # Model Summary
    encoder.summary()
    decoder.summary()

    if args.profile:
        # for disabling internal profiling calls.
        args.profile = False
        eval_time.append(["End to End", (timeit.timeit(lambda : evaluate(input_seq, encoder, decoder),
                            number = args.iterations)), input_seq.shape, predictions.shape])

        for i in range(len(eval_time)):
            eval_time[i][1] = (eval_time[i][1] * 1000 ) / args.iterations

        print(tabulate(eval_time, headers=["Index", "Layer", "Time(ms)", "Input", "Output"], showindex="always", tablefmt='github'))
        args.profile = True

    tf_fim_ops.fim_deinit()

if __name__ == '__main__':
    print('User arguments {}'.format(args))
    gnmt_model_run()
