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

import datetime
import numpy as np
from six.moves import xrange    # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import Model

tf.keras.backend.set_floatx('float16')
initializer = tf.constant_initializer(value=0.1)
profile_layer = True
SEED = 1234

# Supported rnn cells.
SUPPORTED_RNN_LAYERS = {
    "lstm": tf.keras.layers.LSTM,
}

SUPPORTED_RNNS = {
    "lstm": tf.keras.layers.LSTMCell,
}


# Parameters for batch normalization.
_BATCH_NORM_EPSILON = 1e-5
_BATCH_NORM_DECAY = 0.997

# Filters of convolution layer
_CONV_FILTERS = 32


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
#        self._dynamic=training
#        self.training = training
        self.bn = tf.keras.layers.BatchNormalization(
            momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, fused=False, dtype=dtype)

    def call(self, inputs, training):
        return self.bn(inputs=inputs, training=training)


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
#        self._dynamic=training
        self.paddings = [[0, 0], [padding[0], padding[0]],
                         [padding[1], padding[1]], [0, 0]]
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.layer_id = layer_id
#        self.training = training

        self.conv2d = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="valid", use_bias=False,
                                             activation=tf.nn.relu6, name="cnn_{}".format(layer_id), dtype=dtype, kernel_initializer=initializer)
        self.bn = batch_norm(dtype=dtype)  # training = self.training)

    def call(self, inputs, training):
        inputs = tf.pad(tensor=inputs, paddings=self.paddings)
        retval = self.conv2d(inputs=inputs)

        retval = self.bn(retval, training)

        return retval


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

    def __init__(self, rnn_type, rnn_hidden_size, layer_id, is_batch_norm, is_bidirectional, dtype=tf.float16):
        super(rnn_layer, self).__init__()
#        self._dynamic=training
        self.is_batch_norm = is_batch_norm
        self.rnn_type = rnn_type
        self.is_bidirectional = is_bidirectional
        self.float_type = dtype
#        self.training = training
#        self.bn = batch_norm(training = self.training)
        if is_bidirectional:
            rnn_cell = SUPPORTED_RNN_LAYERS[self.rnn_type]
            #initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
            self.fw_rnn = rnn_cell(units=rnn_hidden_size, name="rnn_fw_{}".format(
                layer_id), return_sequences=True, dtype=dtype, kernel_initializer=initializer)
            self.bw_rnn = rnn_cell(units=rnn_hidden_size, name="rnn_bw_{}".format(
                layer_id), go_backwards=True, return_sequences=True, dtype=dtype, kernel_initializer=initializer)
            self.dir_layer = tf.keras.layers.Bidirectional(
                self.fw_rnn, backward_layer=self.bw_rnn, dtype=dtype)
        else:
            rnn_cell = SUPPORTED_RNNS[self.rnn_type]
            self.fw_rnn = rnn_cell(units=rnn_hidden_size, name="rnn_fw_{}".format(
                layer_id), kernel_initializer=initializer)
            self.dir_layer = tf.keras.layers.RNN(self.fw_rnn)

    def call(self, inputs, training):
        print('Rnn called')
        if self.is_batch_norm:
            inputs = batch_norm(self.float_type)(inputs, training)

        if self.is_bidirectional:
            outputs = self.dir_layer(inputs)
            rnn_outputs = tf.concat(outputs, -1)

        else:
            rnn_outputs = self.dir_layer(inputs)
        return rnn_outputs


class DeepSpeech2(tf.keras.Model):
    """Define DeepSpeech2 model."""

    def __init__(self, num_rnn_layers, rnn_type, is_bidirectional,
                 rnn_hidden_size, num_classes, use_bias, dtype=tf.float16):
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
#        self._dynamic=training
        self.num_rnn_layers = num_rnn_layers
        self.rnn_type = rnn_type
        self.is_bidirectional = is_bidirectional
        self.rnn_hidden_size = rnn_hidden_size
        self.num_classes = num_classes
        self.use_bias = use_bias
        self.float_type = dtype

        self.conv_layer_one = conv_bn_layer(padding=(20, 5), filters=_CONV_FILTERS, kernel_size=(
            41, 11), strides=(2, 2), layer_id=1, dtype=self.float_type)
        self.conv_layer_two = conv_bn_layer(padding=(10, 5), filters=_CONV_FILTERS, kernel_size=(
            21, 11), strides=(2, 1), layer_id=2, dtype=self.float_type)
        self.rshape = tf.keras.layers.Reshape((-1, 81*_CONV_FILTERS))
        self.rnn_cell = SUPPORTED_RNNS[self.rnn_type]
        self.lstm = tf.keras.Sequential()
        for layer_counter in xrange(self.num_rnn_layers):
            # No batch normalization on the first layer.
            is_batch_norm = (layer_counter != 0)
            #is_batch_norm = False
            rnn_lyr = rnn_layer(rnn_type=self.rnn_type, rnn_hidden_size=self.rnn_hidden_size,
                                layer_id=layer_counter + 1, is_batch_norm=is_batch_norm, is_bidirectional=self.is_bidirectional, dtype=self.float_type)
#            self.fw_rnn = tf.keras.layers.LSTM(units= rnn_hidden_size, name="rnn_fw_{}".format(layer_counter),return_sequences=True)
#            self.bw_rnn = tf.keras.layers.LSTM(units= rnn_hidden_size, name="rnn_bw_{}".format(layer_counter),go_backwards=True, return_sequences=True)
#            rnn_lyr = tf.keras.layers.Bidirectional(self.fw_rnn, backward_layer=self.bw_rnn)
            self.lstm.add(rnn_lyr)

        self.bnorm = batch_norm(dtype=self.float_type)
        self.dense = tf.keras.layers.Dense(
            self.num_classes, use_bias=self.use_bias, dtype=self.float_type)

    def __call__(self, inputs, training=False):

        start = datetime.datetime.now()

        value = self.conv_layer_one(inputs, training)
        value = self.conv_layer_two(value, training)

        end = datetime.datetime.now()
        duration = end - start
        print('Conv Duration', duration)

        # output of conv_layer2 with the shape of
        # [batch_size (N), times (T), features (F), channels (C)].
        # Convert the conv output to rnn input.
        value = self.rshape(value)
        # RNN layers.

        start = datetime.datetime.now()
        value = self.lstm(value,training)
        end = datetime.datetime.now()
        duration = end - start
        print('Lstm layer Duration: ', duration)


        start = datetime.datetime.now()
        # FC layer with batch norm.
        value = self.bnorm(value, training)
        logits = self.dense(value)
        end = datetime.datetime.now()
        duration = end - start
        print('Fc+bnorm Duration', duration)
        return logits

def profile_ds2_eager(training=False):

     inputs = tf.random.uniform(shape=(4, 282, 161, 1), dtype=tf.float16)

     conv_layer_one = conv_bn_layer(padding=(20, 5), filters=_CONV_FILTERS, kernel_size=(
            41, 11), strides=(2, 2), layer_id=1, dtype=tf.float16)
     conv_layer_two = conv_bn_layer(padding=(10, 5), filters=_CONV_FILTERS, kernel_size=(
            21, 11), strides=(2, 1), layer_id=2, dtype=tf.float16)

     rshape = tf.keras.layers.Reshape((-1, 81*_CONV_FILTERS))
     bn = tf.keras.layers.BatchNormalization(
            momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, fused=False, dtype=tf.float16)

     lstm = tf.keras.Sequential()
     lstm.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1600,
                            kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                            recurrent_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                            return_sequences=True,
                            dtype='float16',
                            trainable=False)))
     for i in range(4):
         lstm.add(bn)
         lstm.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(800,
                            kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                            recurrent_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                            return_sequences=True,
                            dtype='float16',
                            trainable=False)))

     bnorm = batch_norm(dtype=tf.float16)
     dense = tf.keras.layers.Dense(
            29, use_bias=False, dtype=tf.float16)

     for i in range(5):
       start = datetime.datetime.now()
       value = conv_layer_one(inputs, training)
       value = conv_layer_two(value, training)
       end = datetime.datetime.now()
       duration = end - start
       print('Conv Duration:', duration)

       value = rshape(value)
       start = datetime.datetime.now()
       whole_seq_out = lstm(value,training=False)
       end = datetime.datetime.now()
       duration = end - start
       print('Lstm Duration:', duration)

       start = datetime.datetime.now()
       value = bnorm(value, training)
       logits = dense(value)
       end = datetime.datetime.now()
       duration = end - start
       print('Fc+bnorm Duration:', duration)
       return logits

def profile_ds2():
    # if we change to float32 , make sure to changes keras_backend at top of file
    dtype = tf.float16
    num_rnn_layers = 5
    is_bidirectional = True
    use_bias = False
    model = DeepSpeech2(num_rnn_layers, 'lstm',
                        is_bidirectional, 800, 29, use_bias, dtype)

    # Input shape
    x = tf.random.uniform(shape=(4, 282, 161, 1), dtype=dtype)

    print('Runing warmup')
    res = model(x)

    print('Profile Start')
    for i in range(5):
        start = datetime.datetime.now()
        print('Round ', i)
        res = model(x)
        print('out shape',res.shape)
        end = datetime.datetime.now()
        duration = end - start
        print('Duration', duration)

#profile_ds2()
profile_ds2_eager()
