from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

miopen_bn_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('libmiopen_bn.so'))
miopen_bn =  miopen_bn_ops.miopen_bn
