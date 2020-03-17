#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/3/16 9:30
# @Author :llj
"""abcnn模型
参考[1]https://blog.csdn.net/u012526436/article/details/90179466
[2]https://blog.csdn.net/u014793102/article/details/89334875?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task
[]"""
import os
import sys
from load_data import load_char_data,char_index
import pandas as pd
import numpy as np
# import args
from keras import Input,Model
from keras.layers import Embedding,Dense,concatenate,Conv2D,Dropout,LSTM,BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import args
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  #指定运行GPU
# sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
class ABCNN():
    def __init__(self):
        pass
    def laod_data(self,file='./input/test.csv'):
        text1_list, text2_list, label_list = load_char_data(file, data_size=None)
        # print(np.array(text1_list).shape,np.array(text2_list).shape,np.array(label_list).shape)
        # print(text1_list[0],text2_list[0],label_list[0])
        return text1_list, text2_list, label_list
    '''train,process,persist,load'''
    def creat_model(self):
        input_text1=Input(shape=(args.seq_length,),dtype='int32',name="text1")

        embedding_matrix = np.random.rand(args.vocab_size, args.char_embedding_size) # 词嵌入（使用预训练的词向量）
        embed_layer=Embedding(args.vocab_size, args.char_embedding_size, input_length=args.seq_length,weights=[embedding_matrix], trainable=True)
        embed_text1 = embed_layer(input_text1)

        input_text2=Input(shape=(args.seq_length,),dtype='int32',name="text2")
        # embed_layer_2 = Embedding(args.vocab_size, args.char_embedding_size,
        #                           input_length=args.seq_length, weights=[embedding_matrix], trainable=True)
        embed_text2 = embed_layer(input_text2)

        lstm_layer=LSTM(units=256,dropout=0.1)
        lstm_text1=lstm_layer(embed_text1)
        lstm_text2=lstm_layer(embed_text2)

        merged=concatenate([lstm_text1,lstm_text2])
        dense=Dropout(0.1)(merged)
        dense=BatchNormalization()(dense)
        dense=Dense(256,activation='tanh')(dense)
        dense=BatchNormalization()(dense)
        dense=Dropout(0.1)(dense)
        out_pred=Dense(1,activation='sigmoid')(dense)

        model=Model(inputs=[input_text1,input_text2],outputs=out_pred)
        model.compile(optimizer=Adam(lr=1e-3),loss="binary_crossentropy",metrics=['binary_crossentropy','accuracy'])
        return model
    def train(self):
        text1_list, text2_list, label_list=self.laod_data(file='./input/test.csv')
        model=self.creat_model()
        model.summary()
        early_stopping=EarlyStopping(monitor='val_loss',patience=10,verbose=1)
        model.fit(x=[text1_list,text2_list],y=[label_list],epochs=args.epochs,verbose=1,batch_size=args.batch_size,callbacks=[early_stopping])
        # self.persist(model,save_dir="./output/{}_model_weights".format(self.__class__.__name__))
        return model
        pass
    def persist(self,model,save_dir=""):
        self.model_saved_dir = "./output/{}_model_weights.h5".format(self.__class__.__name__)
        model.save(self.model_saved_dir)
        return self.model_saved_dir
    def load(self,model_saved_dir):
        model=self.creat_model()
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
    @classmethod
    def evalute(self,model,test_text1,test_text2,test_label):#模型评估
        score=model.evaluate(x=[test_text1, test_text2],y=test_label,verbose=2,batch_size=64)
        print("Evalute loss={},evalute accuracy={}".format(score[0],score[2]),score)
    def process(self,model,text1,text2):#模型预测
        '''文本处理成词典id序列表示'''
        text1_c_index,text2_c_index=char_index(text1,text2)
        p=model.predict(x=[text1_c_index,text2_c_index])
        print("‘{}’和‘{}’相似的概率为：{}".format(text1[0],text2[0],p[0][0]),p)
        print("‘{}’和‘{}’".format(text1_c_index,text2_c_index))
    def text_vector(self,model,text1,text2):#利用模型将文本表示成向量形式
        text_vec_model=Model(inputs=model.input,outputs=model.get_layer('dense_1').output)
        text1_c_index,text2_c_index=char_index(text1,text2)
        p=text_vec_model.predict(x=[text1_c_index,text2_c_index])
        print("‘{}’和‘{}’".format(text1_c_index,text2_c_index))
        print("向量{}".format(p.shape),p[0].shape)
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()  # 进行配置，使用30%的GPU
config.gpu_options.per_process_gpu_memory_fraction = 0.3
# config.gpu_options.allow_growth = True # 按需要分配GPU
session = tf.Session(config=config)
KTF.set_session(session)  # 设置session
def train_main():
    abcnn = ABCNN()
    abcnn.train()
def evaluate_main():
    abcnn=ABCNN()
    model_saved_dir = "./output/ABCNN_model_weights.h5"
    model = abcnn.load(model_saved_dir)
    test_text1, test_text2, test_label = abcnn.laod_data(file="./input/test.csv")
    abcnn.evalute(model, test_text1, test_text2, test_label)
def predict_main():
    abcnn=ABCNN()
    model=abcnn.load("./output/ABCNN_model_weights.h5")
    text1,text2="你好","你个大傻子"
    while text1!="./stop":
        line=input("请输入text1|text2两个文本（以|分割）:")
        ts=line.split("|")
        if len(ts)==1:
            ts=line.split(',')
        text1,text2=ts[0],ts[1]
        # abcnn.process(model,[text1],[text2])
        abcnn.text_vector(model,[text1],[text2])

if __name__=="__main__":
    predict_main()
