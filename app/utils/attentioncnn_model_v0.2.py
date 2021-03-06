#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/3/25 13:20
# @Author :llj
'''创建分类模型
数据：question.xls label=196类
用input-->multiheadattention-->textcnn->Flatten-->dense
【注】与v0.1区别 基于字符级别的特征
'''
from keras import backend as K
import tensorflow as tf
import time,os,platform,sys
import numpy as np
from data_processor import FeatureRepresentation
from attention_keras import Attention,SelfAttention
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import keras
class Attention_TextCNN_Char_Model():
    '''初始化：词典，预训练词向量矩阵，文本预处理过程'''
    def __init__(self,hyper_parameters):
        word_dictionary_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".." + hyper_parameters['model']['word_dictionary_dir'])
        print(word_dictionary_dir)
        self.fr=FeatureRepresentation(class_num=196,word_dictionary_dir=word_dictionary_dir)
        self.vocab =self.fr.word_dictionary
        self.embedding_dim = self.fr.vector_size
        self.hyper_parameters=hyper_parameters

    def create_model(self,):
        input_text1 = keras.Input(shape=(self.hyper_parameters['seq_length'],), dtype='int32', name="input_text1")
        if self.hyper_parameters['model']['trainable'] is True:
            embedding_matrix = np.random.rand(len(self.vocab) + 1, self.embedding_dim)  # 词嵌入（使用预训练的词向量）
        else:
            embedding_matrix = self.fr.create_embedding_matrix()  # 用预训练的词向量创建embedding
        embedding_layer = keras.layers.Embedding(input_dim=len(self.vocab) + 1, output_dim=self.embedding_dim,
                                                 input_length=self.hyper_parameters['seq_length'],
                                                 weights=[embedding_matrix], trainable=True)

        embedd = embedding_layer(input_text1)#output=(?,sequence_length,embedding_dim)=(None, 64, 300)
        print("embedd",embedd)
        heads=hyper_parameters['model']['heads']
        size_per_head=hyper_parameters['model']['size_per_head']
        embedd = Attention(heads=heads, size_per_head=size_per_head, mask_right=True)([embedd, embedd, embedd])
        print("embedd",embedd)
        embedd_reshape=keras.layers.Reshape((self.hyper_parameters['seq_length'],heads*size_per_head,1))(embedd)
        print("embedd_reshape",embedd_reshape)#(?,sequence_length,embedding_dim,1)=(None, 64, 300, 1)
        # embedd=keras.layers.SpatialDropout1D(0.1)(embedd)
        filter_sizes = [3, 4, 5]
        num_filters = [64, 64, 64]
        conv_pools = []
        for i, filter_size in enumerate(filter_sizes):
            conv = keras.layers.Conv2D(num_filters[i],kernel_size=(filter_size,heads*size_per_head), padding='valid', strides=(1,1),kernel_initializer='normal',activation='relu',name="conv_{}".format(i))(embedd_reshape)
            print("conv",conv)
            pool = keras.layers.MaxPool2D(pool_size=(self.hyper_parameters['seq_length'] - filter_size + 1, 1), strides=(1,1),padding='valid',name='pool_{}'.format(i))(conv)
            print("pool",pool)
            conv_pools.append(pool)
        conv3 = keras.layers.concatenate(conv_pools, axis=-1)
        print(conv3)
        flatten = keras.layers.Flatten()(conv3)

        flatten = keras.layers.Dropout(self.hyper_parameters['model']['dropout'])(flatten)
        dense_1 = keras.layers.Dense(self.hyper_parameters['model']['text_vector_dim'], activation='relu',name='dense_1')(flatten)
        # dense_1=keras.layers.BatchNormalization()(dense_1)
        output = keras.layers.Dense(self.hyper_parameters['class_num'], name='output', activation='softmax')(dense_1)

        model = keras.Model(inputs=input_text1, outputs=output)

        '''调用自定义的facal loss'''
        def focal_loss_fixed(y_true, y_pred, gamma=2., alpha=.25):  # facal loss 为了解决样本分布不均均衡的问题
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
            return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean(
                (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

        # model.compile(optimizer=keras.optimizers.Adam(lr=self.hyper_parameters['model']['lr']),loss=focal_loss_fixed,metrics=[focal_loss_fixed,self.hyper_parameters['model']['metrics']])
        model.compile(optimizer=keras.optimizers.Adam(lr=self.hyper_parameters['model']['lr']),loss="categorical_crossentropy",metrics=["categorical_crossentropy", self.hyper_parameters['model']['metrics']])
        # model.compile(optimizer=keras.optimizers.Adam(lr=self.hyper_parameters['model']['lr']),loss=my_loss,metrics=[my_loss, self.hyper_parameters['model']['metrics']])
        model.summary()
        if (platform.system() == "Windows"):
            from keras.utils import plot_model
            os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38\\bin'
            plot_model(model, to_file="../data/output/{}.png".format(self.__class__.__name__), show_shapes=True,show_layer_names=True)
        self.model = model
        return model

    def train(self, model, train_x, train_y):  # 传入的是处理好的词典id序列
        from keras.callbacks import EarlyStopping
        from keras.callbacks import TensorBoard, ModelCheckpoint
        # early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)
        tensorboard=TensorBoard(log_dir=self.hyper_parameters['model']['log_dir'])
        self.best_model_saved_dir = "../data/output/{}_best_model_weights.h5".format(self.__class__.__name__)
        checkpoint = ModelCheckpoint(filepath=self.best_model_saved_dir, monitor='val_acc', mode='auto', save_best_only='True')
        # model.fit(train_x, train_y,epochs=self.hyper_parameters['model']['epochs'], verbose=1,batch_size=self.hyper_parameters['model']['batch_size'], callbacks=[tensorboard,checkpoint,early_stopping])
        model.fit(train_x, train_y,epochs=self.hyper_parameters['model']['epochs'], verbose=1,batch_size=self.hyper_parameters['model']['batch_size'], callbacks=[tensorboard,checkpoint])
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

import time
def train():
    t0 = time.time()
    train_x, train_y, test_x, test_y = split_data()
    cnn = Attention_TextCNN_Char_Model(hyper_parameters)
    model = cnn.create_model()
    t1 = time.time()
    print("模型创建耗时：{}".format(t1 - t0))
    cnn.train(model, train_x, train_y)
    cnn.evalute(model, test_x, test_y)
    print("训练耗时：{}，训练总耗时：{}".format(time.time() - t1, time.time() - t0))

def predict():
    train_x, train_y, test_x, test_y = split_data()
    print(train_x)
    cnn = Attention_TextCNN_Char_Model(hyper_parameters)
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
    cnn = Attention_TextCNN_Char_Model(hyper_parameters)
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
    similarity_texts()