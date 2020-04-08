#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/4/8 13:57
# @Author :llj

import keras
import keras_bert
import os
'''加载bert模型
参考：https://github.com/yongzhuo/nlp_xiaojiang/blob/master/ClassificationText/bert/keras_bert_classify_bi_lstm.py'''
class KerasBert_Classification():
    def __init__(self,seq_len=128,model_folder='D:\py3.6code\chines-textClassify\chines-TextClassification\\bert-master\chinese_L-12_H-768_A-12',layer_nums=4,training=False,trainable=False,):
        self.model_folder=model_folder
        self.layer_nums = layer_nums
        self.training = training
        self.trainable = trainable
        self.seq_len=seq_len
        self.load()
    def load(self):
        config_path = os.path.join(self.model_folder, 'bert_config.json')
        check_point_path = os.path.join(self.model_folder, 'bert_model.ckpt')
        bert_model = keras_bert.load_trained_model_from_checkpoint(config_path,
                                                                   check_point_path,
                                                               seq_len=self.seq_len,
                                                               output_layer_num=self.layer_nums,
                                                               training=self.training,
                                                               trainable=self.trainable)
        bert_seq_len = int(bert_model.output.shape[1])
        print(bert_seq_len)
        self.bert_model = keras.Model(bert_model.inputs, bert_model.output)
        self.bert_model.summary()

if __name__=="__main__":
    bert_model=KerasBert_Classification().bert_model
