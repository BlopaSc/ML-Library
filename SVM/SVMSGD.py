# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

import numpy as np
import random

def identity(gamma0,**kwargs):
    return gamma0

def decay(gamma0, t, a=None, **kwargs):
    if a is None: a = gamma0
    return gamma0/(1 + (gamma0*t/a))

class SVMSGD():
    def __init__(self, data, descriptor, C=1, lr=1, lr_function=identity, max_iters=10, seed=0, **kwargs):
        self.__prng = random.Random()
        self.__prng.seed(seed)
        self.__C = C
        self.__lr = lr
        self.__lrf = lr_function
        self.__lrkwargs = kwargs
        # Quick access to columns
        self.__columns = list(set(descriptor.get('numerical', descriptor['columns'])) - set([descriptor['target']]))
        self.__columns.sort()
        self.__col_idx = { col : descriptor['columns'].index(col) for col in self.__columns + [descriptor['target']] }
        # Quick access and set of values of the target
        self.__target_idx = self.__col_idx[descriptor['target']]
        self.__it = 0
        self.__data = [[i for i in d] for d in data]
        self.__w = np.array([0 for i in range(len(self.__columns) + 1)])
        self.modify_epochs(max_iters)
    
    def __call__(self, data):
        return [
            (1 if (sum(self.__w[i]*d[self.__col_idx[col]] for i,col in enumerate(self.__columns))+self.__w[-1])>=0 else -1) for d in data
        ]
    
    def modify_epochs(self, iters):
        while self.__it < iters:
            self.__prng.shuffle(self.__data)
            lr = self.__lrf(gamma0=self.__lr, t = self.__it, **self.__lrkwargs)
            for d in self.__data:
                y = d[self.__target_idx]
                x = np.array([d[self.__col_idx[col]] for col in self.__columns] + [1])
                pred = np.dot(self.__w, x)
                if y*pred <= 1:
                    w0 = self.__w.copy()
                    w0[-1] = 0
                    self.__w = self.__w - lr*w0 + lr*self.__C*len(self.__data)*y*x
                else:
                    self.__w[:-1] = (1 - lr)*self.__w[:-1]
            self.__it += 1

    def get_weights(self):
        return [w for w in self.__w]
        
