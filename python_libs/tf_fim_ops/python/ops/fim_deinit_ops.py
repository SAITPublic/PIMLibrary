from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library

fim_deinit_ops = load_library.load_op_library(('libfim_deinit.so'))
fim_deinit = fim_deinit_ops.fim_deinit
