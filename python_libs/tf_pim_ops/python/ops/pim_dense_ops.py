from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

pim_dense_ops = load_library.load_op_library(
    ('libpim_dense.so'))
pim_dense = pim_dense_ops.pim_dense
