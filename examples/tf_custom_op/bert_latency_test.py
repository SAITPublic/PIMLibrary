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
import random

parser = argparse.ArgumentParser(description='Process BERT arguments')
parser.add_argument('-b','--batch_size', default=1, help="Input batch size", type=int)
parser.add_argument('-l','--max_seq_length', default=128, help="Maximum sequence length of BERT input", type=int)
parser.add_argument('-i','--iterations', default=100, help="Number of iterations for profiling", type=int)
parser.add_argument('-p','--profile', action="store_true", help="Enabled/Disable profiling")

args = parser.parse_args()

sample_test_data = [
"This text is included to make sure Unicode is handled properly",
"Text should be one-sentence-per-line, with empty lines between documents.",
"This sample text is public domain and was randomly selected from Project Guttenberg.",
"The rain had only ceased with the gray streaks of morning at Blazing Star, and the settlement awoke to a moral sense of cleanliness, and the finding of forgotten knives, tin cups, and smaller camp utensils, where the heavy showers had washed away the debris and dust heaps before the cabin doors.",
"Indeed, it was recorded in Blazing Star that a fortunate early riser had once picked up on the highway a solid chunk of gold quartz which the rain had freed from its incumbering soil, and washed into immediate and glittering popularity.",
"Possibly this may have been the reason why early risers in that locality, during the rainy season, adopted a thoughtful habit of body, and seldom lifted their eyes to the rifted or india-ink washed skies above them.",
"Cass Beard had risen early that morning, but not with a view to discovery.",
"A leak in his cabin roof,--quite consistent with his careless, improvident habits,--had roused him at 4 A. M., with a flooded bunk and wet blankets.",
"The chips from his wood pile refused to kindle a fire to dry his bed-clothes, and he had recourse to a more provident neighbor's to supply the deficiency.",
"This was nearly opposite.",
"Mr. Cassius crossed the highway, and stopped suddenly.",
"Something glittered in the nearest red pool before him.",
"Gold, surely!",
"But, wonderful to relate, not an irregular, shapeless fragment of crude ore, fresh from Nature's crucible, but a bit of jeweler's handicraft in the form of a plain gold ring.",
"Looking at it more attentively, he saw that it bore the inscription, May to Cass",
"Like most of his fellow gold-seekers, Cass was superstitious.",
"The fountain of classic wisdom, Hypatia herself.",
"As the ancient sage--the name is unimportant to a monk--pumped water nightly that he might study by day, so I, the guardian of cloaks and parasols, at the sacred doors of her lecture-room, imbibe celestial knowledge.",
"From my youth I felt in me a soul above the matter-entangled herd.",
"She revealed to me the glorious fact, that I am a spark of Divinity itself.",
"A fallen star, I am, sir!' continued he, pensively, stroking his lean stomach--'a fallen star!--fallen, if the dignity of philosophy will allow of the simile, among the hogs of the lower world--indeed, even into the hog-bucket itself. Well, after all, I will show you the way to the Archbishop's.",
"There is a philosophic pleasure in opening one's treasures to the modest young.",
"Perhaps you will assist me by carrying this basket of fruit?' And the little man jumped up, put his basket on Philammon's head, and trotted off up a neighbouring street.",
"Philammon followed, half contemptuous, half wondering at what this philosophy might be, which could feed the self-conceit of anything so abject as his ragged little apish guide;",
"but the novel roar and whirl of the street, the perpetual stream of busy faces, the line of curricles, palanquins, laden asses, camels, elephants, which met and passed him, and squeezed him up steps and into doorways, as they threaded their way through the great Moon-gate into the ample street beyond, drove everything from his mind but wondering curiosity, and a vague, helpless dread of that great living wilderness, more terrible than any dead wilderness of sand which he had left behind.",
"Already he longed for the repose, the silence of the Laura--for faces which knew him and smiled upon him; but it was too late to turn back now.",
"His guide held on for more than a mile up the great main street, crossed in the centre of the city, at right angles, by one equally magnificent, at each end of which, miles away, appeared, dim and distant over the heads of the living stream of passengers, the yellow sand-hills of the desert;",
"while at the end of the vista in front of them gleamed the blue harbour, through a network of countless masts.",
"At last they reached the quay at the opposite end of the street;",
"and there burst on Philammon's astonished eyes a vast semicircle of blue sea, ringed with palaces and towers.",
"He stopped involuntarily; and his little guide stopped also, and looked askance at the young monk, to watch the effect which that grand panorama should produce on him."]


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
        random.seed(5)

        input_str_batch    = []
        for x in range(args.batch_size):
            input_str_batch.append(sample_test_data[random.randrange(0,32)])

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
