# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

class VotedPerceptron():
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
        self.__ws = []
        self.__c = 0
        self.modify_epochs(max_iters)
    
    def __predict(self, data, w):
        return [
            (1 if (sum(w[i]*d[self.__col_idx[col]] for i,col in enumerate(self.__columns))+w[-1])>=0 else -1) for d in data
        ]
        
    def __call__(self, data):
        result = [0 for d in data]
        for c,w in self.__ws:
            result = [result[i] + c*v for i,v in enumerate(self.__predict(data, w))]
        return [(1 if r>=0 else -1) for r in result]
    
    def modify_epochs(self, iters):
        while self.__it < iters:
            for d in self.__data:
                y = d[self.__target_idx]
                x = [d[self.__col_idx[col]] for col in self.__columns] + [1]
                pred = sum(self.__w[i]*x[i] for i in range(len(x)))
                # Voted perceptron
                if y*pred <= 0:
                    if self.__ws: self.__ws[-1] = (self.__c, self.__w)
                    self.__w = [self.__w[i] + self.__lr*y*x[i] for i in range(len(x)) ]
                    self.__ws.append((self.__c, self.__w))
                    self.__c = 1
                else:
                    self.__c += 1
            self.__it += 1
        if self.__ws: self.__ws[-1] = (self.__c, self.__w)

    def get_weights(self):
        return [(c,[w for w in ws]) for c,ws in self.__ws]

    def get_distinct_weights(self):
        weights = []
        unique = set()
        for c,ws in self.__ws:
            ws = [round(w,3) for w in ws]
            tws = tuple(ws)
            if tws in unique:
                index = [1 if ws==wi else 0 for ci,wi in weights].find(1)
                weights[index] = (weights[index][0]+c, weights[index][1])
            elif c>0:
                weights.append((c, [w for w in ws]))
                unique.add(tws)
        return weights