# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

import math
import sys
import random

sys.path.append("../DecisionTree")
import ID3

class AdaBoost:
    def __init__(self, data, descriptor, T, treeCls = ID3.ID3, **kwargs):
        self.__treeCls = treeCls
        self.__kwargs = kwargs
        self.__target_idx = descriptor['columns'].index(descriptor['target'])
        self.__vtarget = set([d[self.__target_idx] for d in data])
        # Note: does not handles weighted examples, uses custom weights
        minv = 1/len(data)
        descriptor = {k: descriptor[k] for k in descriptor}
        descriptor['columns'] = [c for c in descriptor['columns']] + ['__ada_weight']
        descriptor['weight'] = '__ada_weight'
        self.__descriptor = descriptor
        self.__data = [[it for it in d]+[minv] for d in data]
        self.__target_idx = descriptor['columns'].index(descriptor['target'])
        self.__weight_idx = len(descriptor['columns'])-1
        self.__classifiers = []
        self.__errors = []
        self.__votes = []
        self.__T = 0
        self.modify_T(T)
        
    def modify_T(self, T):
        while T > self.__T:
            classifier = self.__treeCls(self.__data, self.__descriptor, **self.__kwargs)
            predict = classifier(self.__data)
            error = 0.5 - (sum(d[self.__weight_idx]*(1 if d[self.__target_idx]==predict[i] else -1) for i,d in enumerate(self.__data))/2)
            vote = math.log((1-error)/error)/2
            norm = 0
            for i,d in enumerate(self.__data):
                d[self.__weight_idx] *= math.exp(vote*(-1 if d[self.__target_idx]==predict[i] else 1))
                norm += d[self.__weight_idx]
            for d in self.__data:
                d[self.__weight_idx] /= norm
            # print(sum(data[i][target_idx]==predict[i] for i in range(len(data)) ))
            self.__classifiers.append(classifier)
            self.__errors.append(error)
            self.__votes.append(vote)
            self.__T += 1
        while T < self.__T:
            self.__classifiers.pop()
            self.__errors.pop()
            self.__votes.pop()
            self.__T -= 1
            
    def __call__(self, data):
        votes = [{v: 0 for v in self.__vtarget} for i in range(len(data))]
        for c,classifier in enumerate(self.__classifiers):
            prediction = classifier(data)
            power = self.__votes[c]
            for i in range(len(data)):
                votes[i][prediction[i]] += power
        return [max(zip(value.values(),value.keys()))[1] for value in votes]

    def predict_with_stump(self, data, i):
        return self.__classifiers[i](data)
