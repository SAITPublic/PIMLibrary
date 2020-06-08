from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

fim_bn_ops = load_library.load_op_library(
    ('libfim_bn.so'))
fim_bn = fim_bn_ops.fim_bn
