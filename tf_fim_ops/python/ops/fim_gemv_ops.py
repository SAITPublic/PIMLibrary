from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

fim_gemv_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('libfim_gemv.so'))
fim_gemv = fim_gemv_ops.fim_gemv
