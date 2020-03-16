#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/3/16 10:52
# @Author :llj
import  args
print(args.char_embedding_len)
defaults={"epochs":10,
            "batch_size":64,
              "vocab_size":7901,
              "learning_rate":1e-3,
              "seq_length":15,
              "char_embedding_len":100}
print(defaults)
t="下周有什么好产品？,元月份有哪些理财产品"
print(t.split("|"))