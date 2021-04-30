from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tabulate import tabulate

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import Model
import timeit
import argparse
from tabulate import tabulate
import os
import tf_fim_ops

tf.keras.backend.set_floatx('float16')
"""   [rnnt] config info """

encoder_n_hidden = 1024
encoder_pre_rnn_layers = 2
encoder_stack_time_factor = 2
encoder_post_rnn_layers = 3
pred_n_hidden = 320
pred_rnn_layers = 2
forget_gate_bias = 1.0
joint_n_hidden = 512
dropout = 0.32

# Dense Layer
OUTPUT_SIZE = 29

# Performance table for different layers
eval_time = []

# Parse arguments
parser = argparse.ArgumentParser(description='Process RNNT arguments')
parser.add_argument(
    '-i',
    '--iterations',
    default=100,
    help="Number of iterations for profiling",
    type=int)
parser.add_argument(
    '-p', '--profile', action="store_true", help="Enabled/Disable profiling")
parser.add_argument('-b','--batch_size', default=1, help="Input batch size", type=int)

args = parser.parse_args()


class lstm(tf.keras.layers.Layer):
    def __init__(self, rnn_hidden_size, layer_id):
        super(lstm, self).__init__()

        self.layer_id = layer_id
        self.lstm_layer = layers.LSTM(
            units=rnn_hidden_size,
            return_sequences=True,
            return_state=True,
            dropout=0.32,
            trainable=False,
            recurrent_initializer='random_uniform',
            dtype='float16')

    def call(self, inputs, initial_state=None):

        rnn_outputs, memory_state, carry_state = self.lstm_layer(inputs)
        #print(" LSTM {} input shape {}".format(self.layer_id, inputs.shape))
        if args.profile:
            eval_time.append([
                "LSTM " + str(self.layer_id), (timeit.timeit(
                    lambda: self.lstm_layer(inputs=inputs),
                    number=args.iterations)), inputs.shape, rnn_outputs.shape
            ])

        return rnn_outputs, memory_state, carry_state


class StackTime(tf.keras.layers.Layer):

    __constants__ = ["factor"]

    def __init__(self, factor):
        super(StackTime, self).__init__()
        self.factor = int(factor)

    def call(self, inputs):
        # T, B, U
        x = inputs[0]
        x_lens = inputs[1]
        seq = [x]
        for i in range(1, self.factor):
            # This doesn't seem to make much sense...
            tmp = tf.zeros_like(x)
            """
            This will throw error as item assignment is not supported in tf2.0
            retval = x[i:,:,:]
            print(retval)
            tmp[:-i, :, :] = retval
            """
            print("printing", tmp)
            print("print shape", tmp.shape)

            seq.append(tmp)
        x_lens = tf.math.ceil(x_lens / self.factor)
        print(x_lens)
        # Gross, this is horrible. What a waste of memory...
        return tf.concat(seq, 2)[::self.factor, :, :], x_lens


class RNNT(tf.keras.Model):
    def __init__(self, num_classes):
        super(RNNT, self).__init__()

        self.f = tf.random.uniform([args.batch_size, 1, 1, 1024], dtype=tf.float16)
        self.g = tf.random.uniform([args.batch_size, 1, 1, 320], dtype=tf.float16)
        self.predictor_input = tf.random.uniform([args.batch_size, 1, 320], dtype=tf.float16)
        self.lstm3_input = tf.random.uniform([args.batch_size, 76, 2048], dtype=tf.float16)

        self.encoder_lstm1 = lstm(encoder_n_hidden, 1)
        self.encoder_lstm2 = lstm(encoder_n_hidden, 2)
        #self.stack_time = StackTime(2)
        self.encoder_lstm3 = lstm(encoder_n_hidden, 3)
        self.encoder_lstm4 = lstm(encoder_n_hidden, 4)
        self.encoder_lstm5 = lstm(encoder_n_hidden, 5)
        self.predictor_lstm1 = lstm(pred_n_hidden, 6)
        self.predictor_lstm2 = lstm(pred_n_hidden, 7)
        self.joint_linear1 = keras.layers.Dense(
            joint_n_hidden, activation='relu')
        self.joint_linear2 = keras.layers.Dense(num_classes)

    def call(self, inputs):
        whole_seq_op1, h1, c1 = self.encoder_lstm1(inputs[0])
        whole_seq_op2, h2, c2 = self.encoder_lstm2(whole_seq_op1)
        #x_pad, x_lens =   self.stack_time([whole_seq_op2,inputs[1]])
        #print(x_pad.shape)
        whole_seq_op3, h3, c3 = self.encoder_lstm3(self.lstm3_input)
        whole_seq_op4, h4, c4 = self.encoder_lstm4(whole_seq_op3)
        whole_seq_op5, h5, c5 = self.encoder_lstm5(whole_seq_op4)
        final_encoder_output = tf.transpose(whole_seq_op5)

        pred_op1, hp1, cp1 = self.predictor_lstm1(self.predictor_input)
        pred_op2, hp2, cp2 = self.predictor_lstm2(pred_op1)
        pred_op2 = tf.transpose(pred_op2)
        f = final_encoder_output
        g = pred_op2
        """
           Minor pre-processing before joint- Need proper tf functions for pyorch functions

           B, T, H = f.shape
           B, U_, H2 = g.shape

           f = tf.expand_dims(f,axis=2)   # (B, T, 1, H)
           f = tf.reshape(f,[B, T, U_, H])

           g = tf.expand_dims(g,axis=1)   # (B, 1, U + 1, H)
           g = tf.reshape(g,[B, T, U_, H2])
           """

        joint_ip = tf.concat([self.f, self.g], 3)

        if args.profile:
            eval_time.append([
                "Joint network-Encoder+Predictor output Concat",
                (timeit.timeit(
                    lambda: tf.concat([self.f, self.g], 3),
                    number=args.iterations)), (self.f.shape,
                                               self.g.shape), joint_ip.shape
            ])
        joint_op1 = self.joint_linear1(joint_ip)

        if args.profile:
            eval_time.append([
                "Joint network Linear Layer+Relu", (timeit.timeit(
                    lambda: self.joint_linear1(joint_ip),
                    number=args.iterations)), joint_ip.shape, joint_op1.shape
            ])
        joint_op2 = self.joint_linear2(joint_op1)

        if args.profile:
            eval_time.append([
                "Joint network Linear Layer+Relu", (timeit.timeit(
                    lambda: self.joint_linear2(joint_op1),
                    number=args.iterations)), joint_op1.shape, joint_op2.shape
            ])
        return joint_op2

    def rnnt_e2e(self, inputs):
        whole_seq_op1, h1, c1 = self.encoder_lstm1(inputs[0])
        whole_seq_op2, h2, c2 = self.encoder_lstm2(whole_seq_op1)
        whole_seq_op3, h3, c3 = self.encoder_lstm3(self.lstm3_input)
        whole_seq_op4, h4, c4 = self.encoder_lstm4(whole_seq_op3)
        whole_seq_op5, h5, c5 = self.encoder_lstm5(whole_seq_op4)
        final_encoder_output = tf.transpose(whole_seq_op5)
        pred_op1, hp1, cp1 = self.predictor_lstm1(self.predictor_input)
        pred_op2, hp2, cp2 = self.predictor_lstm2(pred_op1)
        pred_op2 = tf.transpose(pred_op2)
        f = final_encoder_output
        g = pred_op2
        joint_ip = tf.concat([self.f, self.g], 3)

        joint_op1 = self.joint_linear1(joint_ip)

        joint_op2 = self.joint_linear2(joint_op1)

        return joint_op2

def DummyExecute():
    # Create some tensors
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)


if __name__ == '__main__':
    print('User arguments {}'.format(args))
    tf_fim_ops.fim_init()
    rnnt_model = RNNT(OUTPUT_SIZE)
    DummyExecute()

    inputs = tf.random.normal([args.batch_size, 151, 240], dtype=tf.float16)
    output = rnnt_model([inputs, tf.random.normal([args.batch_size, 1], dtype=tf.float16)])

    if args.profile == False:
        end_to_end_latency = timeit.timeit(
            lambda: rnnt_model.rnnt_e2e([inputs, tf.random.normal([1])]),
            number=args.iterations) / args.iterations * 1000
        print("End-to-End RNNT Model Latency (ms)", end_to_end_latency)

    else:
        # Summation of all layers time
        evaltime_sum = sum(row[1] for row in eval_time)
        eval_time.append(["Sum of layers time", evaltime_sum, inputs.shape, output.shape])

        for i in range(len(eval_time)):
            eval_time[i][1] = (eval_time[i][1] * 1000) / args.iterations

        print(tabulate(
            eval_time,
            headers=["Layer", "Time(ms)", "Input Shape", "Output Shape"]))

    tf_fim_ops.fim_deinit()
