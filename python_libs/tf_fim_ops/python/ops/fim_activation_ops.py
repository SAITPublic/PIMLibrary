from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

fim_activation_ops = load_library.load_op_library(
    ('libfim_activation.so'))
fim_act = fim_activation_ops.fim_activation
