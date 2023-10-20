# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

import math
import sys
import random

sys.path.append("../DecisionTree")
import ID3

class Bagging:
    def __init__(self, data, descriptor, T, m=None, seed=None, treeCls = ID3.ID3, **kwargs):
        self.__prng = random.Random()
        self.__prng.seed(seed)
        self.__data = data
        self.__descriptor = descriptor
        self.__m = m if m else len(data)
        self.__treeCls = treeCls
        self.__kwargs = kwargs
        self.__target_idx = descriptor['columns'].index(descriptor['target'])
        self.__vtarget = set([d[self.__target_idx] for d in data])
        
        self.__classifiers = []
        self.__T = 0
        self.modify_T(T)
        
    def modify_T(self, T):
        while T > self.__T:
            data = self.__prng.choices(self.__data, k=self.__m)
            classifier = self.__treeCls(data, self.__descriptor, **self.__kwargs)
            # print(sum(data[i][target_idx]==predict[i] for i in range(len(data)) ))
            self.__classifiers.append(classifier)
            self.__T += 1

    def __call__(self, data):
        votes = [{v: 0 for v in self.__vtarget} for i in range(len(data))]
        for classifier in self.__classifiers:
            prediction = classifier(data)
            for i in range(len(data)):
                votes[i][prediction[i]] += 1
        return [max(zip(value.values(),value.keys()))[1] for value in votes]