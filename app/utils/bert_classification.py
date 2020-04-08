#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/4/8 11:00
# @Author :llj
from keras import backend as K
import tensorflow as tf
import time,os,platform,sys
import numpy as np
from data_processor import FeatureRepresentation
from attention_keras import Attention,SelfAttention
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import kashgari
import keras

'''迁移学习的方法，加载bert模型训练文本分类模型，提取某基层文本的输出作为文本向量计算文本的相似度'''
import os
import shutil
import kashgari
from kashgari.embeddings import BERTEmbedding
import kashgari.tasks.classification as clf
from kashgari.processors import ClassificationProcessor
from sklearn.model_selection import train_test_split
from kashgari.tasks.classification import BLSTMModel
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
class Kashgari_Classification():
    '''初始化：词典，预训练词向量矩阵，文本预处理过程'''
    def __init__(self,hyper_parameters):
        kashgari.config.use_cudnn_cell = False
        processor = ClassificationProcessor(multi_label=False)
        self.bert_embedding = BERTEmbedding(hyper_parameters['model']['bert_model_path'],
                                            task=kashgari.CLASSIFICATION,
                                            layer_nums=hyper_parameters['model']['layer_nums'],
                                            trainable=hyper_parameters['model']['trainable'],
                                            processor=processor,
                                            sequence_length='auto')
        print(len(self.bert_embedding._tokenizer._token_dict_inv))
        self.tokenizer = self.bert_embedding.tokenizer

    def create_model(self,):
        self.bert_embedding.processor.add_bos_eos = False
        model = BLSTMModel(embedding=self.bert_embedding)
        model.fit(valid_x, valid_y, epochs=1)
        res = model.predict(valid_x[:20])
        print(res)
        return model

    def train(self, model, train_x, train_y):  # 传入的是处理好的词典id序列
        from keras.callbacks import EarlyStopping
        from keras.callbacks import TensorBoard, ModelCheckpoint
        # early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)
        tensorboard=TensorBoard(log_dir=self.hyper_parameters['model']['log_dir'])
        self.best_model_saved_dir = "../data/output/{}_best_model_weights.h5".format(self.__class__.__name__)
        checkpoint = ModelCheckpoint(filepath=self.best_model_saved_dir,monitor='val_loss',save_best_only=True,save_weights_only=False,verbose=self.hyper_parameters['model']['verbose'])
        early_stopping = EarlyStopping(monitor='val_loss',patience=self.hyper_parameters['model']['patience'])
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=self.hyper_parameters['model']['factor'],patience=self.hyper_parameters['model']['patience'],verbose=self.hyper_parameters['model']['verbose'])
        model.fit(train_x, train_y,epochs=self.hyper_parameters['model']['epochs'], verbose=1,batch_size=self.hyper_parameters['model']['batch_size'], callbacks=[tensorboard,checkpoint, early_stopping, reduce_lr])
        self.presist(model)
        pass

    def presist(self, model, save_dir=""):
        self.model_saved_dir = "../data/output/{}_model_weights.h5".format(self.__class__.__name__)
        model.save(self.model_saved_dir)
        return self.model_saved_dir

    def load(self, model_saved_dir="../data/output/{}_best_model_weights.h5".format(sys._getframe().f_code.co_name)):
        if os.path.exists(self.hyper_parameters['model']['best_model_saved_dir']):
            model_saved_dir=self.hyper_parameters['model']['best_model_saved_dir']
        else:
            model_saved_dir=self.hyper_parameters['model']['last_model_saved_dir']
        print("加载 model地址:{}".format(model_saved_dir))
        model = self.create_model()
        model.load_weights(model_saved_dir)
        for layer in model.layers:
            print("{}层的权重:{}".format(layer.name, np.array(model.get_layer(layer.name).get_weights()).shape),
                  model.get_layer(layer.name))
            try:
                # print("{}层.input:{},layer.output:{}".format(layer.name,layer.input,layer.output))
                print("{}层.input_shape:{},layer.output_shape:{}".format(layer.name, layer.input_shape,
                                                                        layer.output_shape))
            except AttributeError as e:
                print(layer.get_output_at(0))
                print("AttributeError:{}".format(e))
                continue
        return model

    def evalute(self, model, test_text1, test_label):  # 模型评估,传入的是处理好的词典id序列
        score = model.evaluate(x=test_text1, y=test_label, verbose=2, batch_size=64)
        print("Evalute loss={},evalute accuracy={}".format(score[0], score[2]), score)

    def process(self, model, text1):  # 模型预测，传入文本
        '''文本处理成词典id序列表示'''
        text1 = self.fr.dp.process(text1)
        text1_index = np.array(self.fr.sentence2idx([text1],tokenization=self.fr.dp.char_tokenization))
        p = model.predict(x=text1_index)
        print("文本‘{}’的词典序列{},分类概率{}".format(text1, text1_index, p))

hyper_parameters = {
    "seq_length": 32,  # 文本序列最大长度
    "class_num": 197,  # 类别个数
    "model": {"epochs": 200,
              "dropout": 0.5,
              'lr': 1e-3,  # 学习率,bert取5e-5,其他取1e-3, 对训练会有比较大的影响, 如果准确率一直上不去,可以考虑调这个参数
              "metrics": "accuracy",  # 保存更好模型的评价标准
              "best_model_saved_dir": "../data/output/Attention_TextCNN_Char_Model_best_model_weights.h5",  # 模型权重保存地址
              "last_model_saved_dir": "../data/output/Attention_TextCNN_Char_Model_model_weights.h5",  # 模型权重保存地址
              "word_dictionary_dir": "/data/word_dictionary.json",  # 词典保存地址
              "vector_file": "D:\py3.6code\QA\code\mydgcnn\data\\temp\sgns.wiki.word",  # 预训练词向量地址
              "log_dir":"../data/log",#tensorboard保存路径
              "text_vector_dim": 256,  # 倒数第二层神经元个数
              "batch_size": 64,
              'heads':4,
              'size_per_head':32,
              "trainable": True,
              'layer_nums':4,
              'bert_model_path':"D:\py3.6code\chines-textClassify\chines-TextClassification\\bert-master\chinese_L-12_H-768_A-12",
               "patience": 5,
              "factor": 0.5, # factor of reduce learning late everytime
               "verbose": 1,
              }
}
import random
import pandas as pd
from data_processor import get_config
from collections import Counter

def split_data():#加载数据、处理数据、分割数据
    config=get_config('../conf.ini')
    df = pd.read_excel(config['data']['questions_data'])  # 加载训练数据
    df.drop_duplicates(subset='questions',keep='first', inplace=True)#删除df列中的重复元素
    # print(df.shape, df.columns,df['label'].value_counts())

    class_num = max(dict(Counter(df['label'])).keys()) + 1
    # print(df.head(3),class_num)
    texts, labels = np.array(df['questions']), np.array(df['label'])
    index = [idx for idx in range(len(texts))]
    random.shuffle(index)
    ratio = 0.8
    texts, labels = texts[index], labels[index]
    train_texts, train_labels = texts[:int(ratio * len(texts))], labels[:int(ratio * len(texts))]
    test_texts, test_labels = texts[int(ratio * len(texts)):], labels[int(ratio * len(texts)):]
    print(len(index), len(train_texts), len(test_texts), int(ratio * len(texts)), len(texts))
    word_dictionary_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".." + hyper_parameters['model']['word_dictionary_dir'])
    fr = FeatureRepresentation(class_num=hyper_parameters['class_num'],senquence_len=hyper_parameters['seq_length'],word_dictionary_dir=word_dictionary_dir)
    # dp.process("高速通行费，可以简易征收增值税吗？Can you tell Me雞', '雞', '虎', '牛', '豬', '虎', '兔',chonghong 砖票")
    train_x, train_y = fr.sentence2idx(train_texts,tokenization=fr.dp.char_tokenization), fr.label2onehot(train_labels)
    test_x, test_y = fr.sentence2idx(test_texts,tokenization=fr.dp.char_tokenization),fr.label2onehot(test_labels)
    print(train_x.shape, train_y.shape)
    print(test_x.shape, test_y.shape)
    print("词典大小:{},未来识别词语{}:{}".format(len(fr.word_dictionary),len(fr.OOV),fr.OOV))
    return train_x, train_y, test_x, test_y

sample_train_x = [
    list('语言学（英语：linguistics）是一门关于人类语言的科学研究'),
    list('语言学（英语：linguistics）是一门关于人类语言的科学研究'),
    list('语言学（英语：linguistics）是一门关于人类语言的科学研究'),
    list('语言学包含了几种分支领域。'),
    list('在语言结构（语法）研究与意义（语义与语用）研究之间存在一个重要的主题划分'),
]

sample_train_y = [['b', 'c'], ['a'], ['a', 'c'], ['a', 'b'], ['c']]

sample_eval_x = [
    list('语言学是一门关于人类语言的科学研究。'),
    list('语言学包含了几种分支领域。'),
    list('在语言结构研究与意义研究之间存在一个重要的主题划分。'),
    list('语法中包含了词法，句法以及语音。'),
    list('语音学是语言学的一个相关分支，它涉及到语音与非语音声音的实际属性，以及它们是如何发出与被接收到的。'),
    list('与学习语言不同，语言学是研究所有人类语文发展有关的一门学术科目。'),
    list('在语言结构（语法）研究与意义（语义与语用）研究之间存在一个重要的主题划分'),
]

sample_eval_y = [['b', 'c'], ['a'], ['a', 'c'], ['a', 'b'], ['c'], ['b'], ['a']]

from kashgari.corpus import SMP2018ECDTCorpus
valid_x, valid_y = SMP2018ECDTCorpus.load_data('valid')
print(valid_x[:2])
print(valid_y[:2])

import time
def train():
    t0 = time.time()
    # train_x, train_y, test_x, test_y = split_data()
    cnn = Kashgari_Classification(hyper_parameters)
    model = cnn.create_model()
    model.fit(valid_x, valid_y, epochs=1)
    t1 = time.time()
    print("模型创建耗时：{}".format(t1 - t0))
    # cnn.train(model, train_x, train_y)
    # cnn.evalute(model, test_x, test_y)
    # print("训练耗时：{}，训练总耗时：{}".format(time.time() - t1, time.time() - t0))

def predict():
    train_x, train_y, test_x, test_y = split_data()
    print(train_x)
    cnn = Kashgari_Classification(hyper_parameters)
    model = cnn.load()
    cnn.evalute(model, test_x, test_y)
    text1 = "你好"
    while text1 != "/stop":
        text1 = input(">>>")
        cnn.process(model, text1)

'''将文本转化成向量形式，因为模型中有两个lstm 可以尝试提取两个向量，先计算（同一个文本的）两个向量的距离
不同的文本      计算两个向量的相似度'''
def similarity_vecs_caculate(text1_vec,text2_vec):
    son = np.sum(text1_vec * text2_vec, axis=1)
    mom1 = np.sqrt(np.sum(text1_vec * text1_vec, axis=1))
    mom2 = np.sqrt(np.sum(text2_vec * text2_vec, axis=1))
    scores = son / (mom1 * mom2)
    return scores
    pass
from sklearn.metrics.pairwise import cosine_similarity
def similarity_texts():#讲两个文本表示成向量的形式，计算向量之间的相似度
    cnn = Kashgari_Classification(hyper_parameters)
    model=cnn.load()
    text_vector_model = keras.Model(inputs=model.input, outputs=model.get_layer('dense_1').output)
    text_vector_model.summary()
    fr=FeatureRepresentation(class_num=2)
    texts="你好|你好吗"
    while texts!='./stop':
        texts=input(">>>")
        text1,text2=texts.strip().split('|')
        text1_index = np.array(fr.sentence2idx([fr.dp.process(text1)],tokenization=fr.dp.char_tokenization))  # 文本转化成词典id序列表示
        text2_index = np.array(fr.sentence2idx([fr.dp.process(text2)],tokenization=fr.dp.char_tokenization))  # 文本转化成词典id序列表示
        text1_vec=text_vector_model.predict(x=text1_index)
        text2_vec=text_vector_model.predict(x=text2_index)
        print("文本{}词典id序列{},向量{}".format(text1,text1_index,text1_vec.shape))#这里p[0]\p[1]两个向量一样
        score=similarity_vecs_caculate(text1_vec,text2_vec)
        print("'{}'与'{}'之间cos={}".format(text1,text2,score))

if __name__ == "__main__":
    train()