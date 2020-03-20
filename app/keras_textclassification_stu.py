#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/3/18 9:40
# @Author :llj
# from keras_textclassification import train
from keras_textclassification.data_preprocess.text_preprocess import PreprocessText
from keras_textclassification.m02_TextCNN.graph import TextCNNGraph as Graph
import numpy as np
import pandas as pd
import keras
import random
if __name__=="__main__":
     # 可配置地址
     path_model_dir = 'Y:/tet_keras_textclassification/'
     path_model = path_model_dir + '/textcnn.model'
     path_fineture = path_model_dir + '/fineture.embedding'
     path_hyper_parameters = path_model_dir + '/hyper_parameters.json'

     # 输入训练验证文件地址,sample数据集label填17
     # path_train = path_model_dir + 'data/train.csv'
     # path_valid = path_model_dir + 'data/val.csv'
     # # or 输入训练/预测list, 这时候label选择填3
     path_train = ['游戏,斩 魔仙 者 称号 怎么 得来 的', '文化,我爱你 古文 怎么 说', '健康,牙龈 包住 牙齿 怎么办']
     path_valid = ['娱乐,李克勤 什么 歌 好听', '电脑,UPS 电源 工作 原理', '文化,我爱你 古文 怎么 说 的 呢']
     # 会删除存在的model目录下的所有文件
     # path_model_dir = 'Y:/tet_keras_textclassification/model/'
     hyper_parameters = {
          'len_max': 50,  # 句子最大长度, 固定推荐20-50, bert越长会越慢, 占用空间也会变大, 本地win10-4G设为20就好, 过大小心OOM
          'embed_size': 300,  # 字/词向量维度, bert取768, word取300, char可以更小些
          'vocab_size': 20000,  # 这里随便填的，会根据代码里修改
          'trainable': True,  # embedding是静态的还是动态的, 即控制可不可以微调
          'level_type': 'char',  # 级别, 最小单元, 字/词, 填 'char' or 'word', 注意:word2vec模式下训练语料要首先切好
          'embedding_type': 'random',  # 级别, 嵌入类型, 还可以填'xlnet'、'random'、 'bert'、 'albert' or 'word2vec"
          'gpu_memory_fraction': 0.66,  # gpu使用率
          'model': {'label': 3,  # 类别数
                    'batch_size': 5,  # 批处理尺寸, 感觉原则上越大越好,尤其是样本不均衡的时候, batch_size设置影响比较大
                    'dropout': 0.5,  # 随机失活, 概率
                    'decay_step': 100,  # 学习率衰减step, 每N个step衰减一次
                    'decay_rate': 0.9,  # 学习率衰减系数, 乘法
                    'epochs': 20,  # 训练最大轮次
                    'patience': 3,  # 早停,2-3就好
                    'lr': 5e-5,  # 学习率,bert取5e-5,其他取1e-3, 对训练会有比较大的影响, 如果准确率一直上不去,可以考虑调这个参数
                    'l2': 1e-9,  # l2正则化
                    'activate_classify': 'softmax',  # 最后一个layer, 即分类激活函数
                    'loss': 'categorical_crossentropy',  # 损失函数
                    'metrics': 'accuracy',  # 保存更好模型的评价标准
                    'is_training': True,  # 训练后者是测试模型
                    'model_path': path_model,
                    # 模型地址, loss降低则保存的依据, save_best_only=True, save_weights_only=True
                    'path_hyper_parameters': path_hyper_parameters,  # 模型(包括embedding)，超参数地址,
                    'path_fineture': path_fineture,  # 保存embedding trainable地址, 例如字向量、词向量、bert向量等
                    },
          'embedding': {'layer_indexes': [12],  # bert取的层数
                        # 'corpus_path': '', # embedding预训练数据地址,不配则会默认取conf里边默认的地址
                        },
          'data': {'train_data': path_train,  # 训练数据
                   'val_data': path_valid  # 验证数据
                   },
          'data_path': {'train_data': "keras_textclassification/data/baidu_qa_2019/baike_qa_train.csv",  # 训练数据
                   'val_data': "keras_textclassification/data/baidu_qa_2019/baike_qa_train.csv"  # 验证数据
                   },
     }
     a=np.array([1,2,3,4,5,6,7])
     b=np.array([11,22,33,44,55,66,77])
     index=[i for i in range(0,len(a))]
     random.shuffle(index)
     print(index)
     print(type(a[index].tolist()),a[index],b[index])
     pt=PreprocessText()
     rate=0.1
     graph = Graph(hyper_parameters)
     print("graph init ok!")
     ra_ed = graph.word_embedding
     # print(ra_ed.token2idx,ra_ed.token2idx['[UNK]'])
     x_train, y_train = pt.preprocess_label_ques_to_idx(hyper_parameters['embedding_type'],
                                                        hyper_parameters['data_path']['train_data'],
                                                        ra_ed, rate=rate, shuffle=True)
     print(x_train.shape,x_train[0])
     print(len(np.argmax(y_train,axis=1)),np.argmax(y_train,axis=1).tolist())
     print(graph.word_embedding.input,graph.word_embedding.output)
     model=keras.Model(graph.word_embedding.input,graph.word_embedding.output)
     p=model.predict(x_train)
     print(p.shape,p[0].shape,p[0][0].shape)