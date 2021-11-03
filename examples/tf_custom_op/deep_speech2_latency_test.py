# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Network structure for DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange    # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import Model
import tf_pim_ops
import timeit
import argparse
from tabulate import tabulate
import os
import tf_pim_ops as pim_ops

SEED = 1234

# DeepSpeech2 Configuration parameter
# Input Data layer
INPUT_HEIGHT = 237
INPUT_WIDTH = 171
NUM_CHANNELS = 1

# Filters of convolution layer
CONV_FILTERS = 32
CONV1_KERNEL_H = 41
CONV1_KERNEL_W = 11
CONV2_KERNEL_H = 21
CONV2_KERNEL_W = 11

# Parameters for batch normalization.
BATCH_NORM_EPSILON = 1e-5
BATCH_NORM_DECAY = 0.997

# RNN Layer
NUM_RNN_LAYERS = 5
RNN_HIDDEN_SIZE = 800

# Dense Layer
OUTPUT_SIZE = 29

initializer = tf.keras.initializers.RandomUniform(minval=0, maxval=0.05, seed=SEED)
kernel_initializer = tf.keras.initializers.RandomUniform(minval=0, maxval=0.05, seed=SEED)

# Performance table for different layers
eval_time = []

# Parse arguments
parser = argparse.ArgumentParser(description='Process GNMT arguments')
parser.add_argument('-b','--batch_size', default=1, help="Input batch size", type=int)
parser.add_argument('-l','--max_seq_length', default=50, help="Maximum sequence length of GNMT input", type=int)
parser.add_argument('-i','--iterations', default=100, help="Number of iterations for profiling", type=int)
parser.add_argument('-p','--profile', action="store_true", help="Enabled/Disable profiling")
parser.add_argument('-f','--functional_verify', action="store_true", help="Enabled/Disable Functional verification")
parser.add_argument('-d','--dtype', default='fp16' , help="fp16 or fp32 execution")
parser.add_argument('-m','--module', default='keras' , help="keras or pim_custom execution")

args = parser.parse_args()

class batch_norm(tf.keras.layers.Layer):
    """Batch normalization layer.

    Note that the momentum to use will affect validation accuracy over time.
    Batch norm has different behaviors during training/evaluation. With a large
    momentum, the model takes longer to get a near-accurate estimation of the
    moving mean/variance over the entire training dataset, which means we need
    more iterations to see good evaluation results. If the training data is evenly
    distributed over the feature space, we can also try setting a smaller momentum
    (such as 0.1) to get good evaluation result sooner.

    Args:
    inputs: input data for batch norm layer.
    training: a boolean to indicate if it is in training stage.

    Returns:
    tensor output from batch norm layer.
    """

    def __init__(self, dtype=tf.float16):
        super(batch_norm, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization(
            momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, fused=False, dtype=dtype)

    def call(self, inputs, training):
        bn_out = self.bn(inputs=inputs, training=training)

        if args.profile == True :
            print("batch norm input shape {}".format(inputs.shape))
            eval_time.append(["Batch Normalization",
                (timeit.timeit(lambda: self.bn(inputs=inputs, training=training), number = args.iterations)), inputs.shape, bn_out.shape])

        return bn_out


class conv_bn_layer(tf.keras.layers.Layer):
    """Defines 2D convolutional + batch normalization layer.

    Args:
        inputs: input data for convolution layer.
        padding: padding to be applied before convolution layer.
        filters: an integer, number of output filters in the convolution.
        kernel_size: a tuple specifying the height and width of the 2D convolution
            window.
        strides: a tuple specifying the stride length of the convolution.
        layer_id: an integer specifying the layer index.
        training: a boolean to indicate which stage we are in (training/eval).

    Returns:
        tensor output from the current layer.
    """

    def __init__(self, padding, filters, kernel_size, strides, layer_id, dtype=tf.float16):
        super(conv_bn_layer, self).__init__()

        self.paddings = [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]]
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.layer_id = layer_id

        self.conv2d = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                                strides=strides, padding="valid", use_bias=False,
                                                activation=tf.nn.relu6, name="cnn_{}".format(layer_id),
                                                dtype=dtype, kernel_initializer=initializer)
        self.bn = batch_norm(dtype=dtype)

    def call(self, inputs, training):
        inputs_pad = tf.pad(tensor=inputs, paddings=self.paddings)
        if args.profile == True :
            print(" padding input shape {}".format(inputs.shape))
            eval_time.append(["Padding Conv" + str(self.layer_id),
                (timeit.timeit(lambda: tf.pad(tensor=inputs, paddings=self.paddings), number = args.iterations)), inputs.shape, inputs_pad.shape])

        retval = self.conv2d(inputs=inputs_pad)
        if args.profile == True :
            print(" conv {} input shape {}".format(self.layer_id, inputs.shape))
            eval_time.append(["Convolution " + str(self.layer_id),
                (timeit.timeit(lambda: self.conv2d(inputs=inputs), number = args.iterations)), inputs_pad.shape, retval.shape])

        retval = self.bn(retval, training)

        return retval

class rnn_pim_layer(tf.keras.layers.Layer):
    """Defines a batch normalization + rnn layer.
    Args:
        inputs: input tensors for the current layer.
        rnn_cell: RNN cell instance to use.
        rnn_hidden_size: an integer for the dimensionality of the rnn output space.
        layer_id: an integer for the index of current layer.
        is_batch_norm: a boolean specifying whether to perform batch normalization
            on input states.
        is_bidirectional: a boolean specifying whether the rnn layer is
            bi-directional.
        training: a boolean to indicate which stage we are in (training/eval).

    Returns:
        tensor output for the current layer.
    """

    def __init__(self, rnn_hidden_size, num_layers, is_batch_norm, is_bidirectional, dtype=tf.float16):
        super(rnn_pim_layer, self).__init__()
        self.is_batch_norm = is_batch_norm
        self.float_type = dtype
        self.is_bi_direction = is_bidirectional
        if is_bidirectional :
            self.num_layers = num_layers * 2
            self.bi = 2
        else:
            self.num_layers = num_layers
            self.bi = 1

        #refer ,size_t RNNDescriptor::GetWorkspaceSize()
        self.ws_len = 6 * num_layers * args.batch_size * INPUT_HEIGHT * rnn_hidden_size * 4;
        self.ws_len = self.ws_len *self.bi

        cell_val = 0.0
        hid_val = 0.0
        weight_val = 0.001

        self.hidden_states = tf.constant(hid_val, shape=(1, self.num_layers, args.batch_size, rnn_hidden_size), dtype=tf.float16)
        self.cell_states = tf.constant(cell_val, shape=(1, self.num_layers, args.batch_size, rnn_hidden_size), dtype=tf.float16)

        weight_x = 2592 + ((num_layers - 1) * (self.bi + 1) + 1) * rnn_hidden_size
        weight_y = self.bi * rnn_hidden_size * 4  # nHiddenTensorsPerLayer;
        self.weights_ext = tf.constant(weight_val, shape=(1, weight_x, weight_y), dtype=tf.float16)

    def call(self, inputs):
        inputs = np.expand_dims(inputs, 0)
        result, hidden_out, cell_out, ws = tf_pim_ops.pim_lstm(
                                                            inputs,
                                                            self.weights_ext,
                                                            self.hidden_states,
                                                            self.cell_states,
                                                            tf.constant([self.bi]),
                                                            tf.constant([self.ws_len]))

        rnn_output = np.expand_dims(result[0],0)

        if args.profile == True :
            eval_time.append(["LSTM ",(timeit.timeit(lambda: tf_pim_ops.pim_lstm(
                                            inputs,
                                            self.weights_ext,
                                            self.hidden_states,
                                            self.cell_states,
                                            tf.constant([self.bi]),
                                            tf.constant([self.ws_len]),
                                       ),
                                       number = args.iterations)),
                                       inputs.shape,
                                       rnn_output.shape])

        return rnn_output

class rnn_layer(tf.keras.layers.Layer):
    """Defines a batch normalization + rnn layer.

    Args:
        inputs: input tensors for the current layer.
        rnn_cell: RNN cell instance to use.
        rnn_hidden_size: an integer for the dimensionality of the rnn output space.
        layer_id: an integer for the index of current layer.
        is_batch_norm: a boolean specifying whether to perform batch normalization
            on input states.
        is_bidirectional: a boolean specifying whether the rnn layer is
            bi-directional.
        training: a boolean to indicate which stage we are in (training/eval).

    Returns:
        tensor output for the current layer.
    """

    def __init__(self, rnn_hidden_size, layer_id, is_batch_norm, dtype=tf.float16, initializer = initializer):
        super(rnn_layer, self).__init__()
        self.is_batch_norm = is_batch_norm
        self.float_type = dtype
        self.layer_id = layer_id
        self.lstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units = rnn_hidden_size,
                                kernel_initializer=initializer,
                                recurrent_initializer=initializer,
                                return_sequences=True,
                                dtype=dtype,
                                trainable=False))

    def call(self, inputs, training):
        if self.is_batch_norm:
            inputs = batch_norm(self.float_type)(inputs, training)

        rnn_outputs = self.lstm_layer(inputs)
        if args.profile == True :
            print(" LSTM {} input shape {}".format(self.layer_id, inputs.shape))
            eval_time.append(["LSTM " + str(self.layer_id),
                (timeit.timeit(lambda: self.lstm_layer(inputs=inputs), number = args.iterations)), inputs.shape, rnn_outputs.shape])

        return rnn_outputs

#refer tensorflow/python/keras/layers/recurrent_v2.py
def _canonical_to_params(weights, biases, shape, transpose_weights=False):
    """Utility function convert variable to CuDNN compatible parameter.
    Note that Keras weights for kernels are different from the CuDNN format. Eg.:
    ```
      Keras                 CuDNN
      [[0, 1, 2],  <--->  [[0, 2, 4],
       [3, 4, 5]]          [1, 3, 5]]
    ```
    If the input weights need to be in a unified format, then set
    `transpose_weights=True` to convert the weights.
    Args:
      weights: list of weights for the individual kernels and recurrent kernels.
      biases: list of biases for individual gate.
      shape: the shape for the converted variables that will be feed to CuDNN.
      transpose_weights: boolean, whether to transpose the weights.
    Returns:
      The converted weights that can be feed to CuDNN ops as param.
    """
    def convert(w):
      return tf.transpose(w) if transpose_weights else w

    weights = [tf.reshape(convert(x), shape) for x in weights]
    #biases = [array_ops.reshape(x, shape) for x in biases]
    return tf.concat(weights , axis=0)
    #return tf.concat(weights + biases, axis=0)

def get_params(kernel , recurrent_kernel , bias = None):
    weights = tf.split(kernel, 4, axis=1)
    weights += tf.split(recurrent_kernel, 4, axis=1)
    # CuDNN has an extra set of bias for inputs, we disable them (setting to 0),
    # so that mathematically it is same as the canonical LSTM implementation.
    #full_bias = tf.concat((tf.zeros_like(bias), bias), 0)
    weights = [weights[x] for x in (0, 1, 3, 2, 4, 5, 7, 6)]
    params = _canonical_to_params(
                    weights=weights,
                    biases=None,
                    shape=tf.constant([-1]),
                    transpose_weights=True)
    return params

#Todo: pim.lstm , bias is disabled for now
def get_keras_lstm_weight(model , count , bi_dir=False, add_bias=True):

    for i in range(count):
        layer = model[i]
        lw = layer.get_weights()
        if i==0:
            w = get_params(lw[0] , lw[1] ,  None)
            if bi_dir:
                w_back = get_params(lw[2+add_bias] , lw[3+add_bias] ,  None)
                w = tf.concat([w,w_back] , axis=0)
            else:
                w_f = get_params(lw[0] , lw[1] ,  None)
                w = tf.concat([w,w_f] , axis=0)
                if bi_dir:
                  w_back = get_params(lw[2+add_bias] , lw[3+add_bias] ,  None)
                  w = tf.concat([w,w_back] , axis=0)
    return w

class DeepSpeech2(tf.keras.Model):
    """Define DeepSpeech2 model."""

    def __init__(self, num_rnn_layers, rnn_hidden_size, num_classes, use_bias, dtype=tf.float16):
        """Initialize DeepSpeech2 model.
        Args:
            num_rnn_layers: an integer, the number of rnn layers. By default, it's 5.
            rnn_type: a string, one of the supported rnn cells: gru, rnn and lstm.
            is_bidirectional: a boolean to indicate if the rnn layer is bidirectional.
            rnn_hidden_size: an integer for the number of hidden states in each unit.
            num_classes: an integer, the number of output classes/labels.
            use_bias: a boolean specifying whether to use bias in the last fc layer.
        """
        super(DeepSpeech2, self).__init__()

        # Parameters
        self.num_rnn_layers = num_rnn_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.num_classes = num_classes
        self.use_bias = use_bias
        self.float_type = dtype
        self.lstm = []

        # Layers
        self.input_layer = tf.keras.layers.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, NUM_CHANNELS),
                                                 batch_size=args.batch_size)
        self.conv_layer_one = conv_bn_layer(padding=(0, 0), filters=CONV_FILTERS,
                                            kernel_size=(CONV1_KERNEL_H, CONV1_KERNEL_W),
                                            strides=(2, 2), layer_id=1, dtype=self.float_type)
        self.conv_layer_two = conv_bn_layer(padding=(10, 5), filters=CONV_FILTERS,
                                            kernel_size=(CONV2_KERNEL_H, CONV2_KERNEL_W),
                                            strides=(2, 1), layer_id=2, dtype=self.float_type)
        self.rshape = tf.keras.layers.Reshape((-1, 81*CONV_FILTERS))

        if args.module == 'keras':
            for layer_counter in xrange(self.num_rnn_layers):
                # No batch normalization on the first layer.
                is_batch_norm = (layer_counter != 0)
                self.lstm.append(rnn_layer(rnn_hidden_size=self.rnn_hidden_size,
                                           layer_id=layer_counter + 1, is_batch_norm=is_batch_norm,
                                           dtype=self.float_type))
        else:
            if args.functional_verify:
                for layer_counter in xrange(self.num_rnn_layers):
                    # No batch normalization on the first layer.
                    is_batch_norm = (layer_counter != 0)
                    self.lstm.append(rnn_layer(rnn_hidden_size=self.rnn_hidden_size,
                                               layer_id=layer_counter + 1, is_batch_norm=is_batch_norm,
                                               dtype=self.float_type, initializer=kernel_initializer))

            self.lstm_pim = rnn_pim_layer(rnn_hidden_size=self.rnn_hidden_size,
                                      num_layers = self.num_rnn_layers,
                                      is_batch_norm = True,
                                      is_bidirectional = True,
                                      dtype = dtype
                                      )

        self.bnorm = batch_norm(dtype=self.float_type)
        self.dense = tf.keras.layers.Dense(self.num_classes, use_bias=self.use_bias, dtype=self.float_type)
        self.pim_dense_weights =  tf.random.uniform(shape=(rnn_hidden_size*2 , num_classes), dtype=tf.float16, minval=0, maxval=0.05)
        self.pim_dense_bias = tf.random.uniform(shape=[num_classes], dtype=tf.float16, minval=0, maxval=0.05)
        self.reorder = tf.constant([1])
        self.use_bias_int  = tf.constant([1])
        if self.use_bias == False:
            self.use_bias_int = tf.constant([0])

    def call(self, inputs, training=False):
        # Convolution Layer 1
        conv1 = self.conv_layer_one(inputs, training)

        # Convolution Layer 2
        conv2 = self.conv_layer_two(conv1, training)

        # Reshape
        output = self.rshape(conv2)
        if args.profile == True :
            print(" Reshape input shape {}".format(conv2.shape))
            eval_time.append(["Reshape",
                (timeit.timeit(lambda: self.rshape(inputs=conv2), number = args.iterations)), conv2.shape, output.shape])

        if args.functional_verify:
            orig_env = os.environ['ENABLE_PIM']

            reshape_out_gpu = np.copy(output)
            reshape_out_pim = np.copy(output)
            os.environ['ENABLE_PIM'] = '0'

            for layer_counter in xrange(self.num_rnn_layers):
                reshape_out_gpu = self.lstm[layer_counter](reshape_out_gpu, training)

            os.environ['ENABLE_PIM'] = '1'
            if args.module == 'keras':
                for layer_counter in xrange(self.num_rnn_layers):
                    reshape_out_pim = self.lstm[layer_counter](reshape_out_pim, training)
            else:
                reshape_out_pim = self.lstm_pim(reshape_out_pim)

            os.environ['ENABLE_PIM'] = orig_env

            result = np.testing.assert_array_almost_equal(reshape_out_pim, reshape_out_gpu, decimal=1)
            print("Functional Verification : {}".format(result))

            if orig_env == 1:
                output = reshape_out_pim
            else:
                output = reshape_out_gpu
        else:
            # LSTM Layers
            if args.module == 'keras':
                for layer_counter in xrange(self.num_rnn_layers):
                    output = self.lstm[layer_counter](output, training)
            else:
                output = self.lstm_pim(output)

        # Batch Normalization
        bn_out = self.bnorm(output, training)

        # Dense Layer
        logits = self.dense(bn_out)

        if args.functional_verify:
            #No need to track ENABLE_PIM , since miopen not used for dense.
            weights = self.dense.get_weights()
            bias = self.pim_dense_bias
            if self.use_bias == True:
                    bias = weights[1]
            pim_logits = tf_pim_ops.pim_dense(bn_out, weights[0], bias, self.use_bias_int, self.reorder)
            result = np.testing.assert_array_almost_equal(logits, pim_logits, decimal=1)
        #    tf.test.TestCase.assertAllClose(logits, pim_logits, atol=1e-3)

        if args.profile == True :
            print(" Dense input shape {}".format(bn_out.shape))
            if args.module == 'keras':
                logits = self.dense(bn_out)
                eval_time.append(["Dense",
                    (timeit.timeit(lambda: self.dense(inputs=bn_out), number = args.iterations)), bn_out.shape, logits.shape])
            else:
                logits = tf_pim_ops.pim_dense(bn_out, self.pim_dense_weights, self.pim_dense_bias, self.use_bias_int, self.reorder)
                eval_time.append(["Dense",
                    (timeit.timeit(lambda: tf_pim_ops.pim_dense(bn_out, self.pim_dense_weights, self.pim_dense_bias, self.use_bias_int, self.reorder), number = args.iterations)), bn_out.shape, logits.shape])

            print(" Dense output shape {}".format(logits.shape))
        return logits

def profile_ds2(dtype):
    # if we change to float32 , make sure to changes keras_backend at top of file
    if dtype == tf.float16:
        tf.keras.backend.set_floatx('float16')
    is_bidirectional = True
    use_bias = False

    model = DeepSpeech2(NUM_RNN_LAYERS, RNN_HIDDEN_SIZE, OUTPUT_SIZE, use_bias, dtype)

    # Input shape
    x = tf.random.uniform(shape=(args.batch_size, INPUT_HEIGHT, INPUT_WIDTH, 1), dtype=dtype, minval=0, maxval=0.05)

    if args.profile:
        # For initialization of GPU and PIM preloading
        args.profile = False
        res = model(x)
        args.profile = True

    res = model(x)
    model.summary()

    # Summation of all layers time
    evaltime_sum = sum(row[1] for row in eval_time)
    eval_time.append(["Sum of layers time", evaltime_sum, x.shape, res.shape])

    if args.profile:
        args.profile = False
        eval_time.append(["End to End", timeit.timeit(lambda: model(x), number = args.iterations), x.shape, res.shape])
        args.profile = True

        for i in range(len(eval_time)):
            eval_time[i][1] = (eval_time[i][1] * 1000 ) / args.iterations

        print(tabulate(eval_time, headers=["Index", "Layer", "Time(ms)", "Input Shape", "Output Shape"], showindex="always", tablefmt='github'))

if __name__ == '__main__':
    tf_pim_ops.pim_init()
    print('User arguments {}'.format(args))
    dtype = tf.float16
    if args.dtype == 'fp32':
        dtype = tf.float32
    profile_ds2(dtype)
    tf_pim_ops.pim_deinit()
