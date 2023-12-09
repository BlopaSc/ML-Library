# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

import math
import numpy as np
import random

class NeuralNetwork:
    def __init__(self, data, descriptor, layers, function='s', lr=1, d_decay=0, max_iters=0, seed=None, weights=None):
        # Save layers
        self.p__layers = layers
        self.p__lr = lr
        self.p__d_decay = d_decay if d_decay else lr
        # Quick access to columns
        self.p__columns = list(set(descriptor.get('numerical', descriptor['columns'])) - set([descriptor['target']]))
        self.p__columns.sort()
        self.p__col_idx = { col : descriptor['columns'].index(col) for col in self.p__columns + [descriptor['target']] }
        # Quick access and set of values of the target
        self.p__target_idx = self.p__col_idx[descriptor['target']]
        # Obtain structured data
        self.p__x = np.array([[d[self.p__col_idx[col]] for col in self.p__columns] for d in data]).T
        self.p__y = np.array([d[self.p__target_idx] for d in data])
        # Start random seeded weights
        np.random.seed(seed)
        if weights is None:
            self.p__w = [np.random.randn(layers[i-1]+1, layers[i]) for i in range(1,len(layers))]
        elif type(weights)==str and weights=='zero':
            self.p__w = [np.zeros((layers[i-1]+1, layers[i])) for i in range(1,len(layers))]
        else:
            self.p__w = [np.array(w) for w in weights]
        # Get activation functions and derivatives
        self.p__f, self.p__df = [], []
        for i in range(len(self.p__w)):
            c = function[i%len(function)] 
            if c =='s':
                f = np.vectorize(self.sigmoid)
                df = np.vectorize(self.dsigmoid)
            elif c == 'n':
                f = np.vectorize(self.none)
                df = np.vectorize(self.dnone)
            self.p__f.append(f)
            self.p__df.append(df)
        # Train if needed
        self.zero_grad()
        self.p__it = 0
        self.modify_epochs(max_iters)
        
    def modify_epochs(self, iters):
        while self.p__it < iters:
            L = self.p__x.shape[1]
            np.random.shuffle(self.p__x)
            lr = self.p__lr/(1 + (self.p__lr*self.p__it/self.p__d_decay))
            # SGD
            for i in range(L):
                # lr = self.p__lr/(1 + (self.p__lr*(self.p__it*L+i)/self.p__d_decay))
                self.zero_grad()
                x = self.p__x[:,i]
                y = np.array([self.p__y[i]])
                self.forward(x)
                self.backward(y)
                for i in range(len(self.p__w)):
                    self.p__w[i] -= lr*self.p__dw[i]
            self.p__it += 1
    
    def __call__(self, X):
        X = np.array([[d[self.p__col_idx[col]] for col in self.p__columns] for d in X]).T
        res = self.forward(X)
        return res.flatten()
    
    def zero_grad(self):
        self.p__dw = [np.zeros((self.p__layers[i-1]+1, self.p__layers[i])) for i in range(1,len(self.p__layers))]
        self.p__n = 0
    
    def forward(self, X):
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        L = X.shape[1]
        # Input layer
        self.p__net = [ np.vstack([X, np.ones(L)])]
        self.p__o = [ np.vstack([X, np.ones(L)])]
        for i,w in enumerate(self.p__w):
            # net = wT * o
            # print(w.T.shape, self.p__o[-1].shape)
            calc = np.matmul(w.T, self.p__o[-1])
            self.p__net.append( np.vstack([calc, np.ones(L)]) )
            # o = phi(net)
            calc = self.p__f[i](calc)
            # Add bias
            calc = np.vstack([calc, np.ones(L)])
            self.p__o.append( calc )
        # Return exclude bias
        return self.p__o[-1][:-1]
    
    def backward(self, Y):
        if len(Y.shape) == 1:
            Y = Y[np.newaxis, :]
        self.p__n += Y.shape[1]
        # Error for output layer, we work with phi'(o) instead of phi'(net)
        # delta = 2*(o-y)*phi'(o) 
        self.p__delta = [  ]
        for i in range(len(self.p__w)-1,-1,-1):
            # print(i)
            if i==len(self.p__w)-1:
                # print("Delta:", self.p__o[-1][:-1].shape)
                self.p__delta.insert(0, self.p__o[-1][:-1] - Y)
            else:
                # print("Delta:", self.p__w[i+1].shape, self.p__delta[0].shape)
                self.p__delta.insert(0, np.matmul(self.p__w[i+1], self.p__delta[0])[:-1]  )
            self.p__delta[0] *= self.p__df[i]( self.p__o[i+1][:-1])
            # print("DW:", self.p__o[i].shape, self.p__delta[0].T.shape, self.p__dw[i].shape)
            self.p__dw[i] += np.matmul( self.p__o[i], self.p__delta[0].T )

    def get_gradients(self):
        return [dw.copy() for dw in self.p__dw]
        

    def none(self, x):
        return x
    
    def dnone(self, x):
        return 1

    def sigmoid(self, x):
        return (1/(1 + math.exp(-x))) if -x < 500 else 0
    
    def dsigmoid(self, y):
        return y*(1-y)

