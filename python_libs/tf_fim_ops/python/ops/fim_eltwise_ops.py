from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

fim_eltwise_ops = load_library.load_op_library(
    ('libfim_eltwise.so'))
fim_eltwise = fim_eltwise_ops.fim_eltwise
