from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

pim_gemv_ops = load_library.load_op_library(
    ('libpim_gemv.so'))
pim_gemv = pim_gemv_ops.pim_gemv
