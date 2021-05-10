from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library

pim_init_ops = load_library.load_op_library(('libpim_init.so'))
pim_init = pim_init_ops.pim_init
