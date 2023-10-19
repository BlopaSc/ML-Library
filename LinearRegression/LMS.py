# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

import math
import random

# For analytical weight calculation
import numpy as np

class LMS:
    def __init__(self, data, descriptor, lr, max_iters=0, threshold=1e-6, strategy='batch', seed=None):
        prng = random.Random()
        prng.seed(seed)
        self.__lr = lr
        # Quick access to columns
        self.__columns = list(set(descriptor.get('numerical', descriptor['columns'])) - set([descriptor['target']]))
        self.__columns.sort()
        self.__col_idx = { col : descriptor['columns'].index(col) for col in self.__columns + [descriptor['target']] }
        # Quick access and set of values of the target
        self.__target_idx = self.__col_idx[descriptor['target']]
        
        # Extract data
        self.__x = [[1] + [d[self.__col_idx[col]] for col in self.__columns] for d in data]
        self.__y = [d[self.__target_idx] for d in data]
        # Initilize weights
        self.__w = [(prng.random() if seed else 0) for i in range(len(self.__columns) + 1)]
        self.__error = []
        lr2 = lr*lr
        it = 0
        error_norm = 1
        while not(threshold) or error_norm > threshold:
            if strategy == 'batch':
                x = self.__x
                y = self.__y
                pred = self.__predict(x)
                dif = list(map(lambda x: x[0]-x[1], zip(y,pred)))
                error = sum(d*d for d in dif)/2
            elif strategy == 'stochastic':
                idx = prng.randint(0, len(self.__x)-1)
                x = [self.__x[idx]]
                y = [self.__y[idx]]
                pred = self.__predict(x)
                dif = list(map(lambda x: x[0]-x[1], zip(y,pred)))
                error = sum(d*d for d in list(map(lambda x: x[0]-x[1], zip(self.__y,self.__predict(self.__x)))))/2
            else:
                print("ERROR")
                break
            dw = [
                  -sum(map(lambda x: x[0]*x[1], zip(dif, [d[i] for d in x]))) for i in range(len(self.__w))
            ]
            for i in range(len(self.__w)):
                self.__w[i] -= lr*dw[i]
            self.__error.append(error)
            error_norm = math.sqrt(sum(i*i*lr2 for i in dw))
            it+=1
            if max_iters and it >= max_iters: break
        
        # Analytical weights
        xt = np.matrix(self.__x)
        x = np.matrix(self.__x).transpose()
        xxt = np.matmul(x,xt)
        xxti = np.linalg.inv(xxt)
        y = np.matrix([self.__y]).transpose()
        xy = np.matmul(x, y)
        self.w = np.matmul(xxti, xy)
    
    def __predict(self, x):
        return [ sum( map(lambda x: x[0]*x[1], zip(self.__w, d)) ) for d in x ]
    
    def __call__(self, x):
        return self.__predict([[1] + [d[self.__col_idx[col]] for col in self.__columns] for d in x])
    
    def learning_rate(self):
        return self.__lr
    
    def training_errors(self):
        return [e for e in self.__error]
    
    def weights(self):
        return [w for w in self.__w]
    
    