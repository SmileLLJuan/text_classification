#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/3/16 10:52
# @Author :llj
def func(a,kwargs):
    print(a,kwargs)
kwargs = {'a': 1, 'b': 2, 'c': 3}
func(1,**kwargs)  # {'a': 1, 'b': 2, 'c': 3}