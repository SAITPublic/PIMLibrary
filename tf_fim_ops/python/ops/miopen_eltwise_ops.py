from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

miopen_eltwise_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('libmiopen_eltwise.so'))
miopen_elt = miopen_eltwise_ops.miopen_eltwise

