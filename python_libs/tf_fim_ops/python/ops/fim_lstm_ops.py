from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

fim_lstm_ops = load_library.load_op_library(
    ('libfim_lstm.so'))
fim_lstm = fim_lstm_ops.fim_lstm
