from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library

pim_deinit_ops = load_library.load_op_library(('libpim_deinit.so'))
pim_deinit = pim_deinit_ops.pim_deinit
