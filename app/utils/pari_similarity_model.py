#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/3/25 16:38
# @Author :llj
from keras import backend as K
import tensorflow as tf
import time,os,random,sys
import numpy as np
from data_processor import FeatureRepresentation
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import platform
import keras
'''句子对是否相似的模型
训练分类模型，'''
class PairSimilarity_Model():
    def __init__(self,hyper_parameters):
        self.fr=FeatureRepresentation(class_num=2)
        word_dictionary_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".." + "/data/word_dictionary.json")
        self.vocab = self.fr.load_word_dictionary(word_dictionary_dir=word_dictionary_dir)
        self.vocab_size = self.fr.vector_size
        self.hyper_parameters = hyper_parameters
        self.pos_vocab_size = len(self.fr.pos_dictionary)  # pos向量的维度先设置为pos_dictionary的长度大小

        pass
    def create_model(self):
        input_text1 = keras.Input(shape=(self.hyper_parameters['seq_length'],), dtype='int32', name="input_text1")
        input_text2 = keras.Input(shape=(self.hyper_parameters['seq_length'],), dtype='int32',name="input_text2")
        if self.hyper_parameters['model']['trainable'] is True:
            embedding_matrix = np.random.rand(len(self.vocab) + 1, self.vocab_size)  # 词嵌入（使用预训练的词向量）
        else:
            embedding_matrix = self.fr.create_embedding_matrix()#用预训练的词向量创建embedding
        embedding_layer = keras.layers.Embedding(input_dim=len(self.vocab) + 1, output_dim=self.vocab_size,
                                                 input_length=self.hyper_parameters['seq_length'],
                                                 weights=[embedding_matrix], trainable=True)
        embedded_sequences_1 = embedding_layer(input_text1)
        embedded_sequences_2 = embedding_layer(input_text2)

        # lstm
        lstm_layer = keras.layers.LSTM(self.hyper_parameters['model']['hidden_size'],name="lstm_1")#shar_layer
        lstm_1=lstm_layer(embedded_sequences_1)
        lstm_2=lstm_layer(embedded_sequences_2)
        # lstm_1 = keras.layers.LSTM(self.hyper_parameters['model']['hidden_size'],name="lstm_1")(embedded_sequences_1)
        # lstm_2 = keras.layers.LSTM(self.hyper_parameters['model']['hidden_size'],name="lstm_2")(embedded_sequences_2)

        # classifier
        merged = keras.layers.concatenate([lstm_1, lstm_2])
        merged = keras.layers.Dropout(self.hyper_parameters['model']['dropout'])(merged)
        merged = keras.layers.BatchNormalization()(merged)
        preds = keras.layers.Dense(self.hyper_parameters['class_num'], activation='softmax')(merged)
        model = keras.Model(inputs=[input_text1, input_text2], outputs=preds)
        model.compile(optimizer=keras.optimizers.Adam(lr=self.hyper_parameters['model']['lr']),loss="categorical_crossentropy",metrics=["categorical_crossentropy",self.hyper_parameters['model']['metrics']])
        if (platform.system() == "Windows"):
            from keras.utils import plot_model
            os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38\\bin'
            plot_model(model,to_file="../data/output/{}.png".format(self.__class__.__name__),show_shapes=True,show_layer_names=True)
        self.model=model
        model.summary()
        return model

    def train(self, model, train_x1, train_x2, train_y):  # 传入的是处理好的词典id序列
        from keras.callbacks import EarlyStopping
        from keras.callbacks import TensorBoard, ModelCheckpoint
        early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)
        tensorboard = TensorBoard(log_dir=self.hyper_parameters['model']['log_dir'])
        self.best_model_saved_dir = "../data/output/{}_best_model_weights.h5".format(self.__class__.__name__)
        checkpoint = ModelCheckpoint(filepath=self.best_model_saved_dir, monitor='val_acc', mode='auto',
                                     save_best_only='True')
        model.fit([train_x1, train_x2], train_y, epochs=self.hyper_parameters['model']['epochs'], verbose=1,
                  batch_size=self.hyper_parameters['model']['batch_size'],
                  callbacks=[tensorboard, checkpoint, early_stopping])
        self.presist(model)
        pass

    def presist(self, model, save_dir=""):
        self.model_saved_dir = "../data/output/{}_model_weights.h5".format(self.__class__.__name__)
        model.save(self.model_saved_dir)
        return self.model_saved_dir

    def load(self, model_saved_dir="../data/output/{}_model_weights.h5".format(sys._getframe().f_code.co_name)):
        model= self.create_model()
        model.load_weights(model_saved_dir)
        print(model.get_input_at(0))
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

    def evalute(self, model, test_text1, test_text2, test_label):  # 模型评估,传入的是处理好的词典id序列
        score = model.evaluate(x=[test_text1, test_text2], y=test_label, verbose=2, batch_size=64)
        print("Evalute loss={},evalute accuracy={}".format(score[0], score[2]), score)

    def process(self, model, text1,text2):  # 模型预测，传入文本
        '''文本处理成词典id序列表示'''
        text1 = self.fr.dp.process(text1)
        text2 = self.fr.dp.process(text2)
        text1_index = self.fr.sentence2idx([text1])
        text2_index = self.fr.sentence2idx([text2])
        p = model.predict(x=[text1_index, text2_index])
        print("文本‘{}’的词典序列{},是否相似分类概率{}".format(text1, text1_index, p))

def load_data():#加载句子对 & 创造标签
    with open('../data/zhejiang.csv', 'r', encoding='utf-8') as f:  # 读取 数据
        data = f.readlines()
    ques_dict = {}
    for line in data[1:]:  # 第一行是title
        line = line.split('\t')
        ques_dict[line[0]] = line[3:]  # 前面3个分别对应：标准问题，答案，问题分类

    pool = ques_dict.keys()
    texts_a_list, texts_b_list, labels_list = [], [], []  # 保存所有文本对 & 标签
    for k, v in ques_dict.items():  # 遍历标准问题
        for i in v:  # 遍历k的相似问题
            texts_a_list.append(k)  # 添加标准问题k
            texts_b_list.append(i)  # 添加相似问题i
            labels_list.append(1)
            while True:
                unlike = random.sample(pool, 1)  # 随机选择一个其他类的标准问题
                if unlike[0] != k:
                    break
            texts_a_list.append(k)  # 添加标准问题k
            texts_b_list.append(unlike[0])  # 添加其他类的标注问题（不相似问题）
            labels_list.append(0)
    print(len(texts_a_list), len(texts_b_list), len(labels_list))
    return texts_a_list, texts_b_list, labels_list
def split_data():#加载数据，分割训练数据 & 测试数据
    text1_list, text2_list,label_list = load_data()
    text1s,text2s, labels = np.array(text1_list),np.array(text2_list), np.array(label_list)
    index = [idx for idx in range(len(text1_list))]
    random.shuffle(index)
    ratio=0.8
    text1s,text2s,labels=text1s[index],text2s[index],labels[index]
    train_text1s,train_text2s, train_labels = text1s[:int(ratio*len(text1_list))],text2s[:int(ratio*len(text1_list))], labels[:int(ratio*len(text1_list))]
    test_text1s,test_text2s,test_labels = text1s[int(ratio*len(text1_list)):],text2s[int(ratio*len(text1_list)):], labels[int(ratio*len(text1_list)):]
    print(len(index),len(train_text1s),len(test_text1s),int(ratio*len(text1_list)),len(text1s))

    fr = FeatureRepresentation(class_num=2)
    # dp.process("高速通行费，可以简易征收增值税吗？Can you tell Me雞', '雞', '虎', '牛', '豬', '虎', '兔',chonghong 砖票")
    fr.label2onehot(label_list=label_list)
    train_x1,train_x2,train_y=fr.sentence2idx(train_text1s),fr.sentence2idx(train_text2s),fr.label2onehot(train_labels)
    test_x1,test_x2,test_y=fr.sentence2idx(test_text1s),fr.sentence2idx(test_text2s),fr.label2onehot(test_labels)
    print(train_x1.shape,train_x2.shape,train_y.shape)
    print(test_x1.shape,test_x2.shape,test_y.shape)
    return train_x1,train_x2,train_y,test_x1,test_x2,test_y
hyper_parameters={
    "seq_length":32,#文本序列最大长度
    "class_num":2,#类别个数
    "model":{"epochs":100,
             "dropout":0.1,
             "hidden_size":256,
             'lr': 1e-3,  # 学习率,bert取5e-5,其他取1e-3, 对训练会有比较大的影响, 如果准确率一直上不去,可以考虑调这个参数
             "metrics":"accuracy",  # 保存更好模型的评价标准
             "model_saved_dir":"../data/output/PairSimilarity_Model_best_model_weights.h5",  #模型权重保存地址
             "word_dictionary_dir":"./word_dictionary.json",  #词典保存地址
            "vector_file":"D:\py3.6code\QA\code\mydgcnn\data\\temp\sgns.wiki.word",  #预训练词向量地址
             "log_dir": "../data/log",  # tensorboard保存路径
             "text_vector_dim":256,  #代倒数第二层神经元个数
             "batch_size":256,
             "trainable":True,
             }}
def train():
    t0=time.time()
    train_x1, train_x2, train_y, test_x1, test_x2, test_y = split_data()
    cnn=PairSimilarity_Model(hyper_parameters)
    model = cnn.create_model()
    cnn.train(model,train_x1, train_x2,train_y)
    cnn.evalute(model,test_x1, test_x2, test_y)
    print("训练耗时：{}".format(time.time()-t0))
def predict():
    cnn=PairSimilarity_Model(hyper_parameters)
    model=cnn.load(hyper_parameters['model']['model_saved_dir'])
    texts="你好"
    while texts!='./stop':
        texts=input(">>>")
        text1,text2=texts.strip().split("|")
        cnn.process(model,text1,text2)
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
    cnn = PairSimilarity_Model(hyper_parameters)
    model=cnn.load(hyper_parameters['model']['model_saved_dir'])
    text_vector_model = keras.Model(inputs=model.input, outputs=[model.get_layer('lstm_1').get_output_at(0),model.get_layer('lstm_1').get_output_at(1)])
    text_vector_model.summary()
    fr=FeatureRepresentation(class_num=2)
    texts="你好|你好吗"
    while texts!='./stop':
        texts=input(">>>")
        text1,text2=texts.strip().split('|')
        text1_index = np.array(fr.sentence2idx([fr.dp.process(text1)]))  # 文本转化成词典id序列表示
        text2_index = np.array(fr.sentence2idx([fr.dp.process(text2)]))  # 文本转化成词典id序列表示
        text1_vec=text_vector_model.predict(x=[text1_index,text1_index])[0]
        text2_vec=text_vector_model.predict(x=[text2_index,text2_index])[0]
        print("文本{}词典id序列{},向量{}".format(text1,text1_index,text1_vec.shape))#这里p[0]\p[1]两个向量一样
        score=similarity_vecs_caculate(text1_vec,text2_vec)
        print("'{}'与'{}'之间cos={}".format(text1,text2,score))
if __name__=="__main__":
    similarity_texts()
