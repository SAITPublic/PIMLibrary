from __future__ import absolute_import, division, print_function

import os
import string
import unittest
import tensorflow as tf
import numpy as np

from tensorflow.python import keras
from tabulate import tabulate

from bert import bert_tokenization
import bert
import timeit
import argparse

parser = argparse.ArgumentParser(description='Process BERT arguments')
parser.add_argument('-b','--batch_size', default=1, help="Input batch size", type=int)
parser.add_argument('-l','--max_seq_length', default=128, help="Maximum sequence length of BERT input", type=int)
parser.add_argument('-i','--iterations', default=100, help="Number of iterations for profiling", type=int)
parser.add_argument('-p','--profile', action="store_true", help="Enabled/Disable profiling")

args = parser.parse_args()

class BertTest():

    def setUp(self):
        tf.compat.v1.reset_default_graph()
        keras.backend.clear_session()
        tf.compat.v1.disable_eager_execution()
        print("Eager Execution:", tf.executing_eagerly())

    @staticmethod
    def load_keras_model(model_dir, max_seq_len):
        from tensorflow.python import keras
        from bert import BertModelLayer
        from bert.loader import StockBertConfig, load_stock_weights, params_from_pretrained_ckpt

        bert_config_file = os.path.join(model_dir, "bert_config.json")
        bert_ckpt_file   = os.path.join(model_dir, "bert_model.ckpt")

        l_bert = BertModelLayer.from_params(params_from_pretrained_ckpt(model_dir))

        l_input_ids      = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
        l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="token_type_ids")

        output = l_bert([l_input_ids, l_token_type_ids])

        model = keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=output)
        model.build(input_shape=[(None, max_seq_len),
                                 (None, max_seq_len)])

        load_stock_weights(l_bert, bert_ckpt_file)
        return model

    @staticmethod
    def predict_on_keras_model(model_dir, input_ids, input_mask, token_type_ids):
        max_seq_len = input_ids.shape[-1]

        model = CompareBertActivationsTest.load_keras_model(model_dir, max_seq_len)

        k_res = model.predict([input_ids, token_type_ids])
        return k_res

    def test_bert_latency(self):

        model_name = "uncased_L-12_H-768_A-12"
        model_dir = bert.fetch_google_bert_model(model_name, ".models")
        tokenizer = bert_tokenization.FullTokenizer(vocab_file=os.path.join(model_dir, "vocab.txt"), do_lower_case=True)

        # prepare input
        max_seq_len  = args.max_seq_length
        input_str_batch    = ["hello, bert!", "how are you doing!"]

        input_ids_batch    = []
        token_type_ids_batch = []
        for input_str in input_str_batch:
            input_tokens = tokenizer.tokenize(input_str)
            input_tokens = ["[CLS]"] + input_tokens + ["[SEP]"]

            print("input_tokens len:", len(input_tokens))

            input_ids      = tokenizer.convert_tokens_to_ids(input_tokens)
            input_ids      = input_ids             + [0]*(max_seq_len - len(input_tokens))
            token_type_ids = [0]*len(input_tokens) + [0]*(max_seq_len - len(input_tokens))

            input_ids_batch.append(input_ids)
            token_type_ids_batch.append(token_type_ids)

        input_ids      = np.array(input_ids_batch, dtype=np.int32)
        token_type_ids = np.array(token_type_ids_batch, dtype=np.int32)

        print("   tokens:", input_tokens)
        print("input_ids:{}/{}:{}".format(len(input_tokens), max_seq_len, input_ids), input_ids.shape, token_type_ids)

        model = BertTest.load_keras_model(model_dir, max_seq_len)
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.mean_squared_error)

        pres = model.predict([input_ids, token_type_ids])  # just for fetching the shape of the output
        model.summary()

        # Profiling
        eval_time = []

        eval_time.append(["End to End", timeit.timeit(lambda: model.predict([input_ids, token_type_ids]), number = args.iterations), pres.shape])

        eval_time[0][1] = (eval_time[0][1] * 1000 ) / args.iterations
        print(tabulate(eval_time, headers=["Index", "Layer", "Time(ms)", "Output Shape"], showindex="always", tablefmt='github'))
        print("pres:", pres.shape)


if __name__ == '__main__':
    bert_test = BertTest()
    bert_test.setUp()
    bert_test.test_bert_latency()
