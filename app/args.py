#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/3/16 9:46
# @Author :llj
class_size = 2
char_embedding_len = 100
word_embedding_len = 100

max_char_len = 15
max_word_len = 15
seq_length = 15

batch_size = 200

char_vocab_len = 1692
vocab_size = 7901

learning_rate = 1e-3

keep_prob_ae = 0.8
keep_prob_fully = 0.8
keep_prob_embed = 0.5

keep_prob = 0.7

epochs = 50
lstm_hidden = 100

char_embedding_size = 200
filter_width = 3
filter_height = char_embedding_size
cnn1_filters = 50
cnn2_filters = 50
