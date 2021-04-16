import unittest
import numpy as np
import torch
import torch.nn as nn
import py_fim_ops
import datetime

class PyFimGemvLstmConstant(unittest.TestCase):

    def testLstmLatency(self):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)

        # configs
        miopen_num_layers = num_layers = 5
        bidirectional = True
        hidden_size = 80
        n_batch = 1
        n_seq = 1
        n_input = 25
        n_iters = 10
        ws_len = 96000 * 2 #hack to avoid malloc everytime in custom_op


        bi = 1
        if bidirectional:
            miopen_num_layers = num_layers * 2
            bi = 2

        minv = 0.5
        maxv = 1.0
        inputs = np.random.uniform(minv, maxv, size=[2,n_batch,n_seq,n_input]).astype(np.float16)
        hidden_states = np.random.uniform(minv, maxv, size=[2,miopen_num_layers,inputs.shape[1],hidden_size]).astype(np.float16)
        cell_states = np.random.uniform(minv, maxv, size=[2,miopen_num_layers,inputs.shape[1],hidden_size]).astype(np.float16)

        # Calculate weight dim's referring Miopen driver's /src/rnn.cpp
        weight_x = inputs.shape[3] + \
            ((num_layers  - 1) * (bi + 1) + 1) * hidden_size
        weight_y = bi * hidden_size * 4  # nHiddenTensorsPerLayer;
        weights = np.random.uniform(minv,maxv,size=[2,weight_x,weight_y]).astype(np.float16)

        bi = torch.tensor([bi], dtype=torch.int32).to(device)
        ws_len = torch.tensor([ws_len], dtype=torch.int32).to(device)

        inputs = torch.from_numpy(inputs).to(device)
        weights = torch.from_numpy(weights).to(device)
        hidden_states = torch.from_numpy(hidden_states).to(device)
        cell_states = torch.from_numpy(cell_states).to(device)
        lstm = nn.LSTM(800, 800).cuda()
        #Todo , check with golden
        for i in range(n_iters):
             start = datetime.datetime.now()
             result =  py_fim_ops.py_fim_lstm(
                 inputs, weights, hidden_states, cell_states, bi , ws_len)
             end = datetime.datetime.now()
             duration = end - start
             print('Python Duration:', i , '  ' ,  duration.microseconds/1000)

            #print('Output shape', result.shape)
            #print('Output shape', result.shape)
            #print('Hidden out shape', hidden_out.shape)
            #print('Cell out shape', cell_out.shape)
            #self.assertAllEqual(result, o.reshape(1, out_size))

if __name__ == '__main__':

    py_fim_ops.py_fim_init()
    unittest.main()
    py_fim_ops.py_fim_deinit()
