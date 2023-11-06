# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

class AveragePerceptron():
    def __init__(self, data, descriptor, lr=1, max_iters=10):
        self.__lr = lr
        # Quick access to columns
        self.__columns = list(set(descriptor.get('numerical', descriptor['columns'])) - set([descriptor['target']]))
        self.__columns.sort()
        self.__col_idx = { col : descriptor['columns'].index(col) for col in self.__columns + [descriptor['target']] }
        # Quick access and set of values of the target
        self.__target_idx = self.__col_idx[descriptor['target']]
        self.__it = 0
        self.__data = [[i for i in d] for d in data]
        self.__w = [0 for i in range(len(self.__columns) + 1)]
        self.__a = [0 for i in range(len(self.__columns) + 1)]
        self.modify_epochs(max_iters)
        
    def __call__(self, data):
        return [
            (1 if (sum(self.__a[i]*d[self.__col_idx[col]] for i,col in enumerate(self.__columns))+self.__a[-1])>=0 else -1) for d in data
        ]
    
    def modify_epochs(self, iters):
        while self.__it < iters:
            for d in self.__data:
                y = d[self.__target_idx]
                x = [d[self.__col_idx[col]] for col in self.__columns] + [1]
                pred = sum(self.__w[i]*x[i] for i in range(len(x)))
                # Voted perceptron
                if y*pred <= 0:
                    self.__w = [self.__w[i] + self.__lr*y*x[i] for i in range(len(x)) ]
                self.__a = [self.__a[i] + self.__w[i] for i in range(len(self.__w))]
            self.__it += 1

    def get_weights(self):
        return [w for w in self.__a]