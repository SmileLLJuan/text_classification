#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/3/20 10:59
# @Author :llj
'''在v0.1的版本上添加 词性特征'''

import pymysql
import numpy as np
import pandas as pd
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def get_config(config_dir='./conf.ini'):
    import configparser
    config = configparser.ConfigParser()
    config.read(config_dir)
    print([e for e in config.items()])
    return config
def load_data(config):
    db = pymysql.connect(config['mysql']['ip'], config['mysql']['user'], config['mysql']['password'],
                         config['mysql']['db'], int(config['mysql']['port']))
    table_names=["core_hello","core_industry_ku","core_question_12366"]
    text_list, label_list = [], []
    for table_name in table_names:
        cursor = db.cursor()
        sql = 'select DISTINCT(ques),id from %s' % (table_name)
        print(sql)
        cursor.execute(sql)
        data = cursor.fetchall()
        print(len(data), data[0])
        if table_name=="core_hello":
            for ques,id in data:
                if len(re.findall(u"([\u4e00-\u9fa5A-Za-z0-9@._])", ques))>0:#判断问题中是否包含中文、数字或字符
                    text_list.append(ques)
                    label_list.append(0)#寒喧库的标签设置为0
        elif table_name=="core_industry_ku":
            for ques,id in data:
                if len(re.findall(u"([\u4e00-\u9fa5A-Za-z0-9@._])", ques))>0:#判断问题中是否包含中文、数字或字符
                    text_list.append(ques)
                    label_list.append(1)#行业库的标签设置为0
        elif table_name=="core_question_12366":
            for ques,id in data:
                if len(re.findall(u"([\u4e00-\u9fa5A-Za-z0-9@._])", ques))>0:#判断问题中是否包含中文、数字或字符
                    text_list.append(ques)
                    label_list.append(2)#行业库的标签设置为0
        cursor.close()
    db.close()
    return text_list,label_list
from langconv import Converter
import re,json,os,codecs
import jieba
import jieba.posseg
import gensim


class DataPreprocess():#文本预处理：大写转小写，繁体转简体，删除特殊符号，同义词转化
    def __init__(self):
        self.synonym_list=self.load_synonym(synonym_file="./synonym_dictionary")# 元组最后一个词语是要转化的目标词语，前面的为可能的同义词
        print(self.synonym_list)
        self.senquence_len = 32
        self.class_num=None
        self.word_dictionary=self.load_word_dictionary(word_dictionary_dir="./word_dictionary.json")
            # self.synonym_list = [("红冲", "chonghong", "冲红"), ("人工", "客服", '人工客服'), ("专票", "专用发票"), ("诺诺王", "诺诺网"),("航信", "航天信息")]
        #jiaba 加载停用词 & 词典
        jieba.load_userdict("./jieba_dict.txt")
        self.OOV=[]
        self.vector_size=300
        if os.path.exists("./pos_dictionary.json") is True:
            with open("./pos_dictionary.json",'r',encoding='utf-8') as json_file:
                self.pos_dictionary=json.load(json_file)
        else:
            self.pos_dictionary={'[PAD]': 0, 'ag': 1, 'a': 2, 'ad': 3, 'an': 4, 'b': 5, 'c': 6, 'dg': 7, 'd': 8, 'e': 9, 'f': 10, 'g': 11, 'h': 12, 'i': 13, 'j': 14, 'k': 15, 'l': 16, 'm': 17, 'ng': 18, 'n': 19, 'nr': 20, 'ns': 21, 'nt': 22, 'nz': 23, 'o': 24, 'p': 25, 'q': 26, 'r': 27, 's': 28, 'tg': 29, 't': 30, 'u': 31, 'vg': 32, 'v': 33, 'vd': 34, 'vn': 35, 'w': 36, 'x': 37, 'y': 38, 'z': 39, 'un': 40}

        # self.stopwords = self.load_stopwords(stopword_file='./stopwords.txt')
        pass
    def load_stopwords(self,stopword_file="./stopwords.txt"):
        stopwords = codecs.open(stopword_file, 'r', encoding='utf8').readlines()
        stopwords = [w.strip() for w in stopwords]
        return stopwords
    def load_synonym(self,synonym_file="../data/synonym_dictionary"):  # 加载同义词词典
        with open(synonym_file, 'r', encoding='utf-8') as f:
            data = f.readlines()
        synonym_list = [tuple(line.replace("\n", '').split()) for line in data]
        return synonym_list

    def synonym_replace(self,text="机动车发票怎么冲红,转人工"):#同义词替换 或 错别字替换 或拼音替换
        for synonym in self.synonym_list:
            for i in range(len(synonym) - 1):
                if synonym[i] in text:
                    text = text.replace(synonym[i], synonym[len(synonym) - 1])
                    break
        return text

    def process(self,text):#文本预处理：大写转小写，繁体转简体，删除特殊符号，同义词转化
        text=str(text).lower()#大写转小写
        text = Converter('zh-hans').convert(text)
        text = re.sub('[’!"#$%&\'()*+,-./<=>?@，。/\r\n\t?★、…【】《》？（）“”‘’\\u3000！—[\\]^_`{|}~]+', '', text).replace(" ","")#
        # text = re.sub('[’"#$%&*+-/<=>@/★[\\]^_`{|}~]+', '', text)

        """在这里去掉特殊符号会影响分词效果,所以应该是在分词后在list中去掉特殊符号比较合理,
        eg:诺诺网络科技有限公司->诺诺 网络科技 有限公司
        诺诺网络，科技有限公司->诺诺 网络 ， 科技 有限公司
        但是这种现象出现的概率比较小，先这里删除特殊符号"""
        text=self.synonym_replace(text)
        return text
    def load_word_dictionary(self,word_dictionary_dir="./word_dictionary.json",text_list=None):
        if os.path.exists(word_dictionary_dir) is False:#没有本地字典，需要自己创建
            word_dictionary=self.create_word_dictionary(text_list,word_dictionary_dir)
        else:
            with open(word_dictionary_dir,'r',encoding='utf-8') as json_file:
                word_dictionary=json.load(json_file)
        return word_dictionary
    def create_word_dictionary(self,text_list,word_dictionary_dir="./word_dictionary.json"):#创建字典
        word_dictionary={}
        word_dictionary['[PAD]']=0
        word_dictionary['[UNK]']=1
        word_dictionary['[BOS]']=2
        word_dictionary['[EOS]']=3
        for text in text_list:
            seg_words=jieba.posseg.cut(text)
            for w,p in seg_words:
                if w in word_dictionary:
                    continue
                else:
                    word_dictionary[w]=len(word_dictionary)
        with open(word_dictionary_dir,'w',encoding='utf-8') as json_file:
            json.dump(word_dictionary,json_file,ensure_ascii=False)
        self.word_dictionary=word_dictionary
        return word_dictionary
    def sentence2idx(self,text_list):#senquence_len文本最大长度
        text_words_id_list=[]
        for text in text_list:
            seg_words=jieba.cut(self.process(text),cut_all=False,HMM=True)
            # seg_words=jieba.posseg.cut(self.process(text))
            words=[w for w in seg_words]
            padd_len=self.senquence_len-len(words)
            if padd_len>=0:
                text_words_id=[self.word_dictionary[w] if w in self.word_dictionary else self.word_dictionary['[UNK]'] for w in words]+[self.word_dictionary['[PAD]'] for i in range(padd_len)]
            else:
                text_words_id=[self.word_dictionary[w] if w in self.word_dictionary else self.word_dictionary['[UNK]'] for w in words[0:self.senquence_len]]
            text_words_id_list.append(text_words_id)
        return text_words_id_list
    def label2onehot(self,label_list):
        from collections import Counter
        if self.class_num is None:
            self.class_num=len(Counter(label_list))
        print(Counter(label_list))
        from keras.utils import to_categorical
        label_onehot_list=to_categorical(label_list,num_classes=self.class_num)
        return label_onehot_list
    '''用与预训练的词向量文件创建词典&向量矩阵'''
    def pretrainedVector_create_wordDictionary_embeddingMatrix(self,vector_file="D:\py3.6code\QA\code\mydgcnn\data\\temp\sgns.wiki.word"):
        embedding_model = gensim.models.KeyedVectors.load_word2vec_format(vector_file)
        self.vector_size=embedding_model.wv.vector_size
        word_vectors = embedding_model.wv
        print(len(embedding_model.wv.vocab),embedding_model.wv.vocab['你'])
        print(type(word_vectors))
        print(word_vectors['你'])
        word_dictionary,embedding_matrix={},np.zeros((len(embedding_model.wv.vocab)+4,embedding_model.wv.vector_size))
        word_dictionary['[PAD]'] = 0
        word_dictionary['[UNK]'] = 1
        word_dictionary['[BOS]'] = 2
        word_dictionary['[EOS]'] = 3
        embedding_matrix[0]=np.random.uniform(-0.5,0.5,embedding_model.wv.vector_size)#随机生成向量
        embedding_matrix[1]=np.random.uniform(-0.5,0.5,embedding_model.wv.vector_size)#随机生成向量
        embedding_matrix[2]=np.random.uniform(-0.5,0.5,embedding_model.wv.vector_size)#随机生成向量
        embedding_matrix[3]=np.random.uniform(-0.5,0.5,embedding_model.wv.vector_size)#随机生成向量
        for word,vector in zip(embedding_model.wv.vocab,embedding_model.wv.vectors):
            if '.bin' not in vector_file:
                if word in word_dictionary:
                    continue
                else:
                    word_dictionary[word]=len(word_dictionary)
                embedding_matrix[word_dictionary[word]]=vector
        word_dictionary_dir="./word_dictionary.json"
        with open(word_dictionary_dir,'w',encoding='utf-8') as json_file:
            json.dump(word_dictionary,json_file,ensure_ascii=False)
        embedding_matrix_dir="./embedding_matrix.txt"
        np.savetxt(embedding_matrix_dir, embedding_matrix, encoding='utf-8')
        print("词向量保存路径",embedding_matrix_dir)
        return word_dictionary,embedding_matrix

    def load_pretrainedVector(self,vector_file="./sgns.wiki.word"):
        #vector_file="D:\py3.6code\QA\code\mydgcnn\data\\temp\sgns.wiki.word"
        import gensim
        embedding_model = gensim.models.KeyedVectors.load_word2vec_format(vector_file)
        self.vector_size=embedding_model.wv.vector_size
        word_vectors = embedding_model.wv
        return word_vectors

    def create_embedding_matrix(self,embedding_matrix_dir="./embedding_matrix.txt"):#根据词典生成Embedding层初始化词向量
        if os.path.exists(embedding_matrix_dir) is False:
            word_vectors=self.load_pretrainedVector()
            embedding_matrix = np.zeros((len(self.word_dictionary) + 1, self.vector_size))
            for word,index in self.word_dictionary.items():
                if word in word_vectors:
                    embedding_matrix[index]=word_vectors[word]
                else:
                    embedding_matrix[index]=np.random.uniform(-0.5,0.5,self.vector_size)#未识别的词随机初始化
                    self.OOV.append(word)
            print(embedding_matrix.shape)
            print("未识别的词语",self.OOV)
            np.savetxt(embedding_matrix_dir,embedding_matrix,encoding='utf-8')
        else:
            embedding_matrix=np.loadtxt(embedding_matrix_dir,encoding='utf-8')
        return embedding_matrix
    def pos2id(self,text_list,max_len=32):#文本中的词语分词、提取词性信息，将词性信息转化成特征表示
        text_pos_list=[]
        for text in text_list:
            seg_words = jieba.posseg.cut(text)
            text_pos = []
            for w, p in seg_words:
                if p in self.pos_dictionary:
                    text_pos.append(self.pos_dictionary[p])
                else:
                    self.pos_dictionary[p] = len(self.pos_dictionary)
                    text_pos.append(self.pos_dictionary[p])
            padd_len=max_len-len(text_pos)
            if padd_len>=0:
                text_pos_id=text_pos+[self.pos_dictionary['[PAD]'] for i in range(padd_len)]
            else:
                text_pos_id=text_pos[:max_len]
            text_pos_list.append(text_pos_id)
            with open('./pos_dictionary.json','w',encoding='utf-8') as json_file:
                json.dump(self.pos_dictionary,json_file,ensure_ascii=False)
        return text_pos_list


def split_data():#加载数据，分割训练数据 & 测试数据
    text_list, label_list = load_data(get_config(config_dir='./conf.ini'))
    texts, labels = np.array(text_list), np.array(label_list)
    index = [idx for idx in range(len(text_list))]
    random.shuffle(index)
    ratio=0.8
    texts,labels=texts[index],labels[index]
    train_texts, train_labels = texts[:int(ratio*len(text_list))], labels[:int(ratio*len(text_list))]
    test_texts, test_labels = texts[int(ratio*len(text_list)):], labels[int(ratio*len(text_list)):]
    print(len(index),len(train_texts),len(test_texts),int(ratio*len(text_list)),len(texts))

    dp = DataPreprocess()
    # dp.process("高速通行费，可以简易征收增值税吗？Can you tell Me雞', '雞', '虎', '牛', '豬', '虎', '兔',chonghong 砖票")
    dp.label2onehot(label_list=label_list)
    train_x,train_x_pos,train_y=np.array(dp.sentence2idx(train_texts)),np.array(dp.pos2id(train_texts)),dp.label2onehot(train_labels)
    test_x,test_x_pos,test_y=np.array(dp.sentence2idx(test_texts)),np.array(dp.pos2id(test_texts)),dp.label2onehot(test_labels)
    print(train_x.shape,train_x_pos.shape,train_y.shape)
    print(test_x.shape,test_x_pos.shape,test_y.shape)
    return train_x,train_x_pos,train_y,test_x,test_x_pos,test_y

from keras import backend as K
import tensorflow as tf
import time

import keras
class CNN_POS_Model():#模型中添加了pos特征
    def __init__(self,hyper_parameters):
        dp = DataPreprocess()
        print(len(dp.word_dictionary))
        print(dp.sentence2idx(["你好"]))
        self.vocab =dp.load_word_dictionary(word_dictionary_dir="./word_dictionary.json")
        self.vocab_size=dp.vector_size
        self.dp=dp
        self.hyper_parameters=hyper_parameters
        self.pos_vocab_size=len(dp.pos_dictionary)#pos向量的维度先设置为pos_dictionary的长度大小

        pass
    def create_model(self,):
        input_text1=keras.Input(shape=(self.hyper_parameters['seq_length'],),dtype='int32',name="input_text1")
        input_text1_pos=keras.Input(shape=(self.hyper_parameters['seq_length'],),dtype='int32',name="input_text1_pos")
        embedding_matrix = self.dp.create_embedding_matrix()#用预训练的词向量创建embedding
        # embedding_matrix = np.random.rand(len(self.vocab)+1, self.vocab_size) # 词嵌入（使用预训练的词向量）
        embedding_matrix_pos = np.random.rand(len(self.dp.pos_dictionary)+1, self.pos_vocab_size) # pos词嵌入
        print(embedding_matrix.shape)
        embedd=keras.layers.Embedding(input_dim=len(self.vocab)+1,output_dim=self.vocab_size,input_length=self.hyper_parameters['seq_length'],weights=[embedding_matrix], trainable=self.hyper_parameters['model']['trainable'])(input_text1)
        embedd_pos=keras.layers.Embedding(input_dim=len(self.dp.pos_dictionary)+1,output_dim=self.pos_vocab_size,input_length=self.hyper_parameters['seq_length'],weights=[embedding_matrix_pos], trainable=True)(input_text1_pos)
        embedd=keras.layers.SpatialDropout1D(0.1)(embedd)
        embedd_pos=keras.layers.SpatialDropout1D(0.1)(embedd_pos)
        embedd=keras.layers.concatenate([embedd,embedd_pos])
        lstm_layer=keras.layers.LSTM(units=256,dropout=0.3,recurrent_dropout=0.3)(embedd)
        dense_1=keras.layers.Dense(units=1024,activation='relu',name='dense_1')(lstm_layer)
        dense_1=keras.layers.Dropout(self.hyper_parameters['model']['dropout'])(dense_1)
        dense_2=keras.layers.Dense(self.hyper_parameters['model']['text_vector_dim'],activation='relu',name='dense_2')(dense_1)
        dense_2=keras.layers.Dropout(self.hyper_parameters['model']['dropout'])(dense_2)
        output=keras.layers.Dense(self.hyper_parameters['class_num'],name='output',activation='softmax')(dense_2)
        model=keras.Model(inputs=[input_text1,input_text1_pos],outputs=output)
        '''调用自定义的facal loss'''

        def focal_loss_fixed( y_true, y_pred, gamma=2., alpha=.25):  # facal loss 为了解决样本分布不均均衡的问题
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
            return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean(
                (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
        # model.compile(optimizer=keras.optimizers.Adam(lr=self.hyper_parameters['model']['lr']),loss=focal_loss_fixed,metrics=[focal_loss_fixed,self.hyper_parameters['model']['metrics']])
        model.compile(optimizer=keras.optimizers.Adam(lr=self.hyper_parameters['model']['lr']),loss="categorical_crossentropy",metrics=["categorical_crossentropy",self.hyper_parameters['model']['metrics']])
        model.summary()
        self.model=model
        return model

    def train(self,model,train_x,train_x_pos, train_y):#传入的是处理好的词典id序列
        from keras.callbacks import EarlyStopping
        early_stopping=EarlyStopping(monitor='loss',patience=10,verbose=1)
        model.fit([train_x,train_x_pos], train_y,epochs=self.hyper_parameters['model']['epochs'],verbose=1,batch_size=self.hyper_parameters['model']['batch_size'],callbacks=[early_stopping])
        self.presist(model)
        pass

    def presist(self,model,save_dir=""):
        self.model_saved_dir = "./output/{}_model_weights.h5".format(self.__class__.__name__)
        model.save(self.model_saved_dir)
        return self.model_saved_dir

    def load(self,model_saved_dir):
        model=self.create_model()
        model.load_weights(model_saved_dir)
        for layer in model.layers:
            print("{}层的权重:{}".format(layer.name,np.array(model.get_layer(layer.name).get_weights()).shape),model.get_layer(layer.name))
            try:
                # print("{}层.input:{},layer.output:{}".format(layer.name,layer.input,layer.output))
                print("{}层.input_shape:{},layer.output_shape:{}".format(layer.name,layer.input_shape,layer.output_shape))
            except AttributeError as e:
                print(layer.get_output_at(0))
                print("AttributeError:{}".format(e))
                continue
        model.summary()
        return model

    def evalute(self, model, test_text1,test_text1_pos, test_label):  # 模型评估,传入的是处理好的词典id序列
        score = model.evaluate(x=[test_text1,test_text1_pos], y=test_label, verbose=2, batch_size=64)
        print("Evalute loss={},evalute accuracy={}".format(score[0], score[2]), score)

    def process(self, model, text1):  # 模型预测，传入文本
        '''文本处理成词典id序列表示'''
        text1=self.dp.process(text1)
        text1_index= np.array(self.dp.sentence2idx([text1]))
        text1_pos_index=np.array(self.dp.pos2id([text1]))
        p = model.predict(x=[text1_index,text1_pos_index])
        print("文本‘{}’的词典序列{},分类概率{}".format(text1,text1_index,p))

    def text_vector(self,model,text1):#将文本表示成向量形式
        text_vector_model=keras.Model(inputs=model.inputs,outputs=model.get_layer('dense_2').output)
        text1 = self.dp.process(text1)
        text1_pos_index=np.array(self.dp.pos2id([text1]))
        text1_index=np.array(self.dp.sentence2idx([text1]))#文本转化成词典id序列表示
        p=text_vector_model.predict(x=[text1_index,text1_pos_index])
        print("文本{}词典id序列{},向量{}".format(text1,text1_index,p))

hyper_parameters={
    "seq_length":32,#文本序列最大长度
    "class_num":3,#类别个数
    "model":{"epochs":100,
             "dropout":0.1,
             'lr': 1e-3,  # 学习率,bert取5e-5,其他取1e-3, 对训练会有比较大的影响, 如果准确率一直上不去,可以考虑调这个参数
             "metrics":"accuracy",# 保存更好模型的评价标准
             "model_saved_dir":"./output/CNN_POS_Model_model_weights.h5",#模型权重保存地址
             "word_dictionary_dir":"./word_dictionary.json",#词典保存地址
            "vector_file":"D:\py3.6code\QA\code\mydgcnn\data\\temp\sgns.wiki.word",#预训练词向量地址
             "text_vector_dim":256,#代倒数第二层神经元个数
             "batch_size":256,
             "trainable":False,
             }
}
def train():
    t0=time.time()
    train_x, train_x_pos, train_y, test_x, test_x_pos, test_y = split_data()
    cnn=CNN_POS_Model(hyper_parameters)
    model = cnn.create_model()
    cnn.train(model,train_x, train_x_pos,train_y)
    cnn.evalute(model,test_x,test_x_pos, test_y)
    print("训练耗时：{}".format(time.time()-t0))
def predict():
    cnn=CNN_POS_Model(hyper_parameters)
    model = cnn.create_model()
    cnn.load(hyper_parameters['model']['model_saved_dir'])
    text1="你好"
    while text1!='./stop':
        text1=input(">>>")
        cnn.process(model,text1)
if __name__=="__main__":
    predict()

