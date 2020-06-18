from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library

fim_init_ops = load_library.load_op_library(('libfim_init.so'))
fim_init = fim_init_ops.fim_init
