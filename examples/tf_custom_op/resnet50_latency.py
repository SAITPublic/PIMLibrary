# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=invalid-name
"""ResNet models for Keras.

Reference:
  - [Deep Residual Learning for Image Recognition](
      https://arxiv.org/abs/1512.03385) (CVPR 2015)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras import layers
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.util.tf_export import keras_export

import timeit
import argparse
import os
from tabulate import tabulate

import tf_pim_ops
from tf_pim_ops.python.layers import pim_layers

parser = argparse.ArgumentParser(description='Process Resnet arguments')
parser.add_argument('-b','--batch_size', default=1, help="Input batch size", type=int)
parser.add_argument('-i','--iterations', default=10, help="Number of iterations for profiling", type=int)
parser.add_argument('-p','--profile', action="store_true", help="Enabled/Disable profiling")
parser.add_argument('-d','--dtype', default='fp16' , help="fp16 or fp32 execution")
parser.add_argument('-m','--module', default='keras' , help="keras or pim execution")

args = parser.parse_args()

def ResNet(stack_fn,
           preact,
           use_bias,
           model_name='resnet',
           include_top=True,
           weights='imagenet',
           input_tensor=None,
           input_shape=None,
           pooling=None,
           classes=1000,
           classifier_activation='softmax',
           **kwargs):
  """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.

  Reference:
  - [Deep Residual Learning for Image Recognition](
      https://arxiv.org/abs/1512.03385) (CVPR 2015)

  Optionally loads weights pre-trained on ImageNet.
  Note that the data format convention used by the model is
  the one specified in your Keras config at `~/.keras/keras.json`.

  Arguments:
    stack_fn: a function that returns output tensor for the
      stacked residual blocks.
    preact: whether to use pre-activation or not
      (True for ResNetV2, False for ResNet and ResNeXt).
    use_bias: whether to use biases for convolutional layers or not
      (True for ResNet and ResNetV2, False for ResNeXt).
    model_name: string, model name.
    include_top: whether to include the fully-connected
      layer at the top of the network.
    weights: one of `None` (random initialization),
      'imagenet' (pre-training on ImageNet),
      or the path to the weights file to be loaded.
    input_tensor: optional Keras tensor
      (i.e. output of `layers.Input()`)
      to use as image input for the model.
    input_shape: optional shape tuple, only to be specified
      if `include_top` is False (otherwise the input shape
      has to be `(224, 224, 3)` (with `channels_last` data format)
      or `(3, 224, 224)` (with `channels_first` data format).
      It should have exactly 3 inputs channels.
    pooling: optional pooling mode for feature extraction
      when `include_top` is `False`.
      - `None` means that the output of the model will be
          the 4D tensor output of the
          last convolutional layer.
      - `avg` means that global average pooling
          will be applied to the output of the
          last convolutional layer, and thus
          the output of the model will be a 2D tensor.
      - `max` means that global max pooling will
          be applied.
    classes: optional number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified.
    classifier_activation: A `str` or callable. The activation function to use
      on the "top" layer. Ignored unless `include_top=True`. Set
      `classifier_activation=None` to return the logits of the "top" layer.
    **kwargs: For backwards compatibility only.
  Returns:
    A `keras.Model` instance.

  Raises:
    ValueError: in case of invalid argument for `weights`,
      or invalid input shape.
    ValueError: if `classifier_activation` is not `softmax` or `None` when
      using a pretrained top layer.
  """
  
  if kwargs:
    raise ValueError('Unknown argument(s): %s' % (kwargs,))
  if not (weights in {'imagenet', None} or file_io.file_exists_v2(weights)):
    raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization), `imagenet` '
                     '(pre-training on ImageNet), '
                     'or the path to the weights file to be loaded.')

  if weights == 'imagenet' and include_top and classes != 1000:
    raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                     ' as true, `classes` should be 1000')

  # Determine proper input shape
  input_shape = imagenet_utils.obtain_input_shape(
      input_shape,
      default_size=224,
      min_size=32,
      data_format=backend.image_data_format(),
      require_flatten=include_top,
      weights=weights)

  if input_tensor is None:
    img_input = layers.Input(shape=input_shape)
  else:
    if not backend.is_keras_tensor(input_tensor):
      img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor

  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

  x = layers.ZeroPadding2D(
      padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
  x = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

  if not preact:
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
    x = layers.Activation('relu', name='conv1_relu')(x)

  x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
  x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

  x = stack_fn(x)

  if preact:
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='post_bn')(x)
    x = layers.Activation('relu', name='post_relu')(x)

  if include_top:
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    imagenet_utils.validate_activation(classifier_activation, weights)
    if args.module == 'pim':
        x = pim_layers.PimDense(classes, activation=classifier_activation,
                                name='predictions')(x)
    else:
        x = layers.Dense(classes, activation=classifier_activation,
                         name='predictions')(x)
  else:
    if pooling == 'avg':
      x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
      x = layers.GlobalMaxPooling2D(name='max_pool')(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = layer_utils.get_source_inputs(input_tensor)
  else:
    inputs = img_input

  # Create model.
  model = training.Model(inputs, x, name=model_name)

  return model


def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
  """A residual block.

  Arguments:
    x: input tensor.
    filters: integer, filters of the bottleneck layer.
    kernel_size: default 3, kernel size of the bottleneck layer.
    stride: default 1, stride of the first layer.
    conv_shortcut: default True, use convolution shortcut if True,
        otherwise identity shortcut.
    name: string, block label.

  Returns:
    Output tensor for the residual block.
  """
  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

  if conv_shortcut:
    shortcut = layers.Conv2D(
        4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
  else:
    shortcut = x

  x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
  x = layers.Activation('relu', name=name + '_1_relu')(x)

  x = layers.Conv2D(
      filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
  x = layers.Activation('relu', name=name + '_2_relu')(x)

  x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

  x = layers.Add(name=name + '_add')([shortcut, x])
  x = layers.Activation('relu', name=name + '_out')(x)
  return x


def stack1(x, filters, blocks, stride1=2, name=None):
  """A set of stacked residual blocks.

  Arguments:
    x: input tensor.
    filters: integer, filters of the bottleneck layer in a block.
    blocks: integer, blocks in the stacked blocks.
    stride1: default 2, stride of the first layer in the first block.
    name: string, stack label.

  Returns:
    Output tensor for the stacked blocks.
  """
  x = block1(x, filters, stride=stride1, name=name + '_block1')
  for i in range(2, blocks + 1):
    x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
  return x


def block2(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
  """A residual block.

  Arguments:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default False, use convolution shortcut if True,
        otherwise identity shortcut.
      name: string, block label.

  Returns:
    Output tensor for the residual block.
  """
  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

  preact = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_preact_bn')(x)
  preact = layers.Activation('relu', name=name + '_preact_relu')(preact)

  if conv_shortcut:
    shortcut = layers.Conv2D(
        4 * filters, 1, strides=stride, name=name + '_0_conv')(preact)
  else:
    shortcut = layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

  x = layers.Conv2D(
      filters, 1, strides=1, use_bias=False, name=name + '_1_conv')(preact)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
  x = layers.Activation('relu', name=name + '_1_relu')(x)

  x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
  x = layers.Conv2D(
      filters,
      kernel_size,
      strides=stride,
      use_bias=False,
      name=name + '_2_conv')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
  x = layers.Activation('relu', name=name + '_2_relu')(x)

  x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
  x = layers.Add(name=name + '_out')([shortcut, x])
  return x


def stack2(x, filters, blocks, stride1=2, name=None):
  """A set of stacked residual blocks.

  Arguments:
      x: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, stack label.

  Returns:
      Output tensor for the stacked blocks.
  """
  x = block2(x, filters, conv_shortcut=True, name=name + '_block1')
  for i in range(2, blocks):
    x = block2(x, filters, name=name + '_block' + str(i))
  x = block2(x, filters, stride=stride1, name=name + '_block' + str(blocks))
  return x


def block3(x,
           filters,
           kernel_size=3,
           stride=1,
           groups=32,
           conv_shortcut=True,
           name=None):
  """A residual block.

  Arguments:
    x: input tensor.
    filters: integer, filters of the bottleneck layer.
    kernel_size: default 3, kernel size of the bottleneck layer.
    stride: default 1, stride of the first layer.
    groups: default 32, group size for grouped convolution.
    conv_shortcut: default True, use convolution shortcut if True,
        otherwise identity shortcut.
    name: string, block label.

  Returns:
    Output tensor for the residual block.
  """
  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

  if conv_shortcut:
    shortcut = layers.Conv2D(
        (64 // groups) * filters,
        1,
        strides=stride,
        use_bias=False,
        name=name + '_0_conv')(x)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
  else:
    shortcut = x

  x = layers.Conv2D(filters, 1, use_bias=False, name=name + '_1_conv')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
  x = layers.Activation('relu', name=name + '_1_relu')(x)

  c = filters // groups
  x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
  x = layers.DepthwiseConv2D(
      kernel_size,
      strides=stride,
      depth_multiplier=c,
      use_bias=False,
      name=name + '_2_conv')(x)
  x_shape = backend.int_shape(x)[1:-1]
  x = layers.Reshape(x_shape + (groups, c, c))(x)
  x = layers.Lambda(
      lambda x: sum(x[:, :, :, :, i] for i in range(c)),
      name=name + '_2_reduce')(x)
  x = layers.Reshape(x_shape + (filters,))(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
  x = layers.Activation('relu', name=name + '_2_relu')(x)

  x = layers.Conv2D(
      (64 // groups) * filters, 1, use_bias=False, name=name + '_3_conv')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

  x = layers.Add(name=name + '_add')([shortcut, x])
  x = layers.Activation('relu', name=name + '_out')(x)
  return x


def stack3(x, filters, blocks, stride1=2, groups=32, name=None):
  """A set of stacked residual blocks.

  Arguments:
    x: input tensor.
    filters: integer, filters of the bottleneck layer in a block.
    blocks: integer, blocks in the stacked blocks.
    stride1: default 2, stride of the first layer in the first block.
    groups: default 32, group size for grouped convolution.
    name: string, stack label.

  Returns:
    Output tensor for the stacked blocks.
  """
  x = block3(x, filters, stride=stride1, groups=groups, name=name + '_block1')
  for i in range(2, blocks + 1):
    x = block3(
        x,
        filters,
        groups=groups,
        conv_shortcut=False,
        name=name + '_block' + str(i))
  return x


@keras_export('keras.applications.resnet50.ResNet50',
              'keras.applications.resnet.ResNet50',
              'keras.applications.ResNet50')
def ResNet50(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
  """Instantiates the ResNet50 architecture."""

  def stack_fn(x):
    x = stack1(x, 64, 3, stride1=1, name='conv2')
    x = stack1(x, 128, 4, name='conv3')
    x = stack1(x, 256, 6, name='conv4')
    return stack1(x, 512, 3, name='conv5')

  return ResNet(stack_fn, False, True, 'resnet50', include_top, weights,
                input_tensor, input_shape, pooling, classes, **kwargs)





SEED = 1235
initializer = tf.keras.initializers.RandomNormal(seed=SEED)

eval_time = []

def DummyExecute():
    # Create some tensors
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)

def resnet_model_run(dtype):
    image_width = 224
    image_height = 224
    channels = 3

    input = tf.random.uniform(shape=(args.batch_size, image_height, image_width, channels),dtype=dtype)
    model = ResNet50()

    if args.profile:
        # model and gpu initialization and LUT load
        DummyExecute()
        args.profile = False
        predictions = model(input,training=False)
        args.profile = True

    eval_time.clear()
    predictions = model(input,training=False)

    #Model Summary , Todo:
    model.summary()

    if args.profile:
        # for disabling internal profiling calls.
        args.profile = False
        eval_time.append(["End to End", (timeit.timeit(lambda : model(input,training=False),
                            number = args.iterations)), input.shape, predictions.shape])

        for i in range(len(eval_time)):
            eval_time[i][1] = (eval_time[i][1] * 1000 ) / args.iterations

        print(tabulate(eval_time, headers=["Index", "Layer", "Time(ms)", "Input", "Output"], showindex="always", tablefmt='github'))
        args.profile = True

if __name__ == '__main__':
    print('User arguments {}'.format(args))
    tf_pim_ops.pim_init()
    dtype = tf.float16
    if args.dtype == 'fp32':
        dtype = tf.float32
    else:
        tf.keras.backend.set_floatx('float16')
    resnet_model_run(dtype)
    tf_pim_ops.pim_deinit()
