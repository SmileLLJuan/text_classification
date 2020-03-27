#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/3/25 9:46
# @Author :llj
import logging
logging.basicConfig(level=logging.DEBUG,format='LINE %(lineno)-4d  %(levelname)-8s %(message)s',datefmt='%m-%d %H:%M',
    filename="./log/BlogNetease.log",filemode='w',)
import configparser
config = configparser.ConfigParser()
config.read('conf.ini')
print(config.items(),config['score']['output_num'])
'''【步骤】
1、用户输入，文本处理 大小写转化、同义词替换，纠错'''
import sys
sys.path.append('./utils')
from utils.data_processor import DataPreprocess,FeatureRepresentation
class IntelligenceServices():
    def __init__(self):
        print("你好")
        fr=FeatureRepresentation()
        train_texts=['诺诺网络科技有限公司******<><，。[]{}【】？？？？','niaho ','n你是谁']
        train_labels=[0,2,1]
        train_x, train_x_pos, train_y = fr.sentence2idx(train_texts), fr.pos2id(train_texts), fr.label2onehot(
            train_labels)
        print(train_x, train_x_pos, train_y)
        pass
if __name__=="__main__":
    IS=IntelligenceServices()