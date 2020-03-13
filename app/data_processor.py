#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/3/12 10:06
# @Author :llj
'''处理数据
1、样本数据生成
2、数据处理
3、创建batch_generator'''
import pickle,re
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
class GenerateSamples():#读取数据，生成样本
    def __init__(self):
        pass
    def get_data_dict(self,file_path='../data/zhejiang.csv'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
        ques_dict = {}#key=标准问题,value=[相似问题list]
        for line in data[1:]:
            line = line.split('\t')
            ques_dict[line[0]] = line[3:]
        return ques_dict
    def sample(self,ques_dict,n=10):
        # ques_dict = {1:[2, 3, 4, 5],"a":["b","c","d"],'你':['好','明','天']}
        pos_list, neg_list = [], []
        all_list = []  # 将pos_list & neg_list合并程一个三元组形式，（问题1，问题2，问题3）问题1、2相似，与问题3不相似
        pool = ques_dict.keys()
        for k, a in ques_dict.items():
            for i in range(len(a)):
                pos_list.append([k, a[i]])
                candidate_neg_list = []
                while True:
                    minus = random.sample(pool, 1)  # 随机选择一个标准问题
                    if minus[0] != k:
                        candidate_neg_list.append(minus[0])
                        candidate_neg_list.extend(ques_dict[minus[0]])
                        index = random.randint(0, len(candidate_neg_list) - 1)
                        break
                neg_list.append([k, candidate_neg_list[index]])
                all_list.append([k, a[i], candidate_neg_list[index]])
                for j in range(i + 1, len(a)):
                    pos_list.append([a[i], a[j]])
                    candidate_neg_list = []
                    while True:
                        minus = random.sample(pool, 1)  # 随机选择一个标准问题
                        if minus[0] != k:
                            candidate_neg_list.append(minus[0])
                            candidate_neg_list.extend(ques_dict[minus[0]])
                            index = random.randint(0, len(candidate_neg_list) - 1)
                            break
                    neg_list.append([a[i], candidate_neg_list[index]])  # 随机选择一个标准问题下的相似问题
                    all_list.append([a[i], a[j], candidate_neg_list[index]])
        return pos_list,neg_list
    def create_samples(self):#生成文本分类正负类样本
        ques_dict = self.get_data_dict()
        pos_list, neg_list = self.sample(ques_dict, 10)
        text1_list,text2_list,label_list=[],[],[]
        for text1,text2 in pos_list:
            text1_list.append(text1)
            text2_list.append(text2)
            label_list.append(1)
        for text1,text2 in neg_list:
            text1_list.append(text1)
            text2_list.append(text2)
            label_list.append(0)
        df_data=pd.DataFrame({"text1":text1_list,"text2":text2_list,"label":label_list})
        df_data=df_data.sample(frac=1)#打乱df的顺序，返回比率=1
        # print(df_data.head(2))
        return df_data
class DataLoader():
    def __init__(self):#创建data generator
        with open('../data/char_dictionary.pkl', 'rb') as f:# 加载字符级别的词典
            self.char_dictionary = pickle.load(f)
        self.OOV=[]#保存未识别的字符
        gs = GenerateSamples()
        df_data = gs.create_samples()
        ratio = 0.8
        df_data_train = df_data[:int(len(df_data) * ratio)]
        df_data_test = df_data[int(len(df_data) * ratio):]

        # print(self.char_dictionary)

        self.train_text1 = np.array([self.text2id(text1) for text1 in df_data_train['text1']])
        self.train_text2 = np.array([self.text2id(text1) for text1 in df_data_train['text2']])
        self.train_label = df_data_train['label']
        self.test_text1 = np.array([self.text2id(text1) for text1 in df_data_test['text1']])
        self.test_text2 = np.array([self.text2id(text1) for text1 in df_data_test['text2']])
        self.test_label = df_data_test['label']
        print(self.train_text1.shape,self.train_text2.shape,len(self.train_label))
        print(self.test_text1.shape,self.test_text2.shape,len(self.test_label))

    def text2id(self,text,max_len=64):#将文本转化成词典id序列表示
        text = re.sub('[’!"#$%&\'()*+,-./<=>?@，。/\r\n\t?★、…【】《》？（）“”‘’！[\\]^_`{|}~]+', '', text)
        charid_list=[]
        for char in text:
            if char in self.char_dictionary:
                charid_list.append(self.char_dictionary[char])
                # print("char",char,charid_list)
            else:
                self.OOV.append(char)
        if len(charid_list)<max_len:#填充序列长度
            charid_list.extend([0 for i in range(max_len-len(charid_list))])
            padd_charid_list=charid_list
        else:
            padd_charid_list=charid_list[:max_len]
        return padd_charid_list
    def get_batch(self,batch_size): # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, np.shape(self.train_text1)[0], batch_size)
        print(index)
        return self.train_text1[index],self.train_text2[index], self.train_label[index]
if __name__=="__main__":
    dl=DataLoader()
    from collections import Counter
    counter = Counter(dl.test_label)
    print(counter)
    print(dl.train_label)
    train_text1, train_text2, train_label = dl.train_text1, dl.train_text2, dl.train_label
    test_text1, test_text2, test_label = dl.test_text1, dl.test_text2, dl.test_label
    for i in range(len(train_label[:10])):
        print(type(train_text2[i]), train_text1[i].shape, train_text2[i].shape, train_label[i])
    # train_batch_text1,train_batch_text2,train_batch_label=dl.get_batch(2)
    # print(train_batch_text1,train_batch_text2,train_batch_label)