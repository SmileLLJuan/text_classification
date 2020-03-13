#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/3/12 14:31
# @Author :llj
import pickle
import numpy as np
import keras.layers
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  #指定运行GPU
class MLP():
    def __init__(self):
        super().__init__()
    def creat_model(self):
        with open('../data/char_dictionary.pkl', 'rb') as f:  # 加载字符级别的词典
            char_dictionary = pickle.load(f)
        print(len(char_dictionary))
        embedd_dim = 300
        num_labels = 1
        input_length = max_len = 64
        # 模型结构：词嵌入-卷积池化*3-拼接-全连接-dropout-全连接
        def text_cnn():
            main_input = keras.Input(shape=(input_length,), dtype='float64')
            embedding_matrix = np.random.rand(len(char_dictionary)+1, embedd_dim)
            # 词嵌入（使用预训练的词向量）
            embedder = keras.layers.Embedding(len(char_dictionary) + 1, embedd_dim, input_length=input_length, weights=[embedding_matrix], trainable=True)
            embed = embedder(main_input)
            # 词窗大小分别为3,4,5
            cnn1 = keras.layers.Convolution1D(256, 3, padding='same', strides=1, activation='relu')(embed)
            cnn1 = keras.layers.MaxPool1D(pool_size=4)(cnn1)
            cnn2 = keras.layers.Convolution1D(256, 4, padding='same', strides=1, activation='relu')(embed)
            cnn2 = keras.layers.MaxPool1D(pool_size=4)(cnn2)
            cnn3 = keras.layers.Convolution1D(256, 5, padding='same', strides=1, activation='relu')(embed)
            cnn3 = keras.layers.MaxPool1D(pool_size=4)(cnn3)
            # 合并三个模型的输出向量
            cnn = keras.layers.concatenate([cnn1, cnn2, cnn3], axis=-1)
            return main_input,cnn
        input_1,cnn1=text_cnn()
        input_2, cnn2 = text_cnn()
        cnn = keras.layers.concatenate([cnn1, cnn2], axis=-1)
        flat = keras.layers.Flatten()(cnn)
        drop = keras.layers.Dropout(0.2)(flat)
        main_output = keras.layers.Dense(num_labels, activation='softmax')(drop)
        self.model = keras.Model(inputs=[input_1,input_2], outputs=main_output)
        self.model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    def train(self,train_text1,train_text2,train_label):
        self.model.fit(x=[train_text1,train_text2],y=[train_label],epochs=10,verbose=1,batch_size=256)
from data_processor import DataLoader
if __name__=="__main__":
    batch_size=32
    num_epochs=10
    learning_rate = 0.001
    dl = DataLoader()
    train_text1,train_text2,train_label=dl.train_text1,dl.train_text2,dl.train_label
    test_text1,test_text2,test_label=dl.test_text1,dl.test_text2,dl.test_label
    # for i in range(len(train_label[:10])):
    #     print(type(train_text2[i]),train_text1[i].shape,train_text2[i].shape,train_label[i])

    import tensorflow as tf
    import keras.backend.tensorflow_backend as KTF
    config = tf.ConfigProto()  # 进行配置，使用30%的GPU
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    # config.gpu_options.allow_growth = True # 按需要分配GPU
    session = tf.Session(config=config)
    KTF.set_session(session)  # 设置session
    m=MLP()
    m.creat_model()
    m.train(train_text1,train_text2,train_label)

