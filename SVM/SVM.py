# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

import numpy as np
from scipy.optimize import minimize

class SVM():
    def __init__(self, data, descriptor, C=1, kernel='none', **kwargs):
        self.__C = C
        # Quick access to columns
        self.__columns = list(set(descriptor.get('numerical', descriptor['columns'])) - set([descriptor['target']]))
        self.__columns.sort()
        self.__col_idx = { col : descriptor['columns'].index(col) for col in self.__columns + [descriptor['target']] }
        # Quick access and set of values of the target
        self.__target_idx = self.__col_idx[descriptor['target']]
        # Obtain structured data
        self.__x = np.array([[d[self.__col_idx[col]] for col in self.__columns] for d in data])
        self.__y = np.array([d[self.__target_idx] for d in data])
        
        def no_kernel(x1, x2, **kwargs):
            return np.dot(x1, x2)
        
        def gaussian_kernel(x1, x2, gamma, **kwargs):
            return np.exp(-(np.linalg.norm(x1 - x2.T)**2)/gamma)
        
        if kernel=='none':
            self.__kernel = no_kernel
        elif kernel == 'gaussian':
            self.__kernel = gaussian_kernel
        self.__kernel_kwargs = kwargs
        
        # Target function un-optimized: took > 30 minutes so I just shot execution down
        def target(x,y,alpha):
            result = 0
            for i in range(len(x)):
                for j in range(len(x)):
                    result += y[i]*y[j]*alpha[i]*alpha[j]*np.dot(x[i], x[j])
            return -result/2 + alpha.sum()
        
        # Optimized version, equivalent to the previous double for sum, precalculate M cause stays constant
        M = np.matmul(self.__x * self.__y[:, np.newaxis], (self.__x * self.__y[:, np.newaxis]).T)
        def target_opt(M, alpha):
            return -alpha.dot(alpha.dot(M))/2 + alpha.sum()
        
        # Optimized version 2, which supports kernel function, precalculate phi_xx matrix because constant, tried to add the yy in precalculation but results were being weird
        n = len(self.__x)
        phi_xx = np.array([[self.__kernel(self.__x[i], self.__x[j], **self.__kernel_kwargs) for j in range(n)] for i in range(n)])
        # phi_xxyy = np.dot(self.__y, np.dot(self.__y, phi_xx)) # Didnt worked for some reason @@
        def target_optk(phi_xx, y, alpha):
            return np.dot(alpha*y, np.dot(alpha*y, phi_xx))/2 - np.sum(alpha)
        
        
        # Initial alpha zeros
        alpha = np.zeros(len(self.__x))
        # Bounds for alpha (min, max), 0 < alpha < C
        bounds = [(0, self.__C) for i in range(len(self.__x))]
        # Constraints all alpha*y == 0 ---> alpha dot y == 0
        def constraint(y, alpha):
            return np.dot(y, alpha)
        
        # result = minimize(fun=lambda a: -target(self.__x, self.__y, a), x0=alpha, bounds=bounds, constraints=[{'type': 'eq', 'fun': lambda a: constraint(self.__y, a)}])
        # result = minimize(fun=lambda a: -target_opt(M, a), x0=alpha, bounds=bounds, constraints=[{'type': 'eq', 'fun': lambda a: constraint(self.__y, a)}])
        result = minimize(fun=lambda a: target_optk(phi_xx, self.__y, a), x0=alpha, bounds=bounds, constraints=[{'type': 'eq', 'fun': lambda a: constraint(self.__y, a)}])
        
        self.__alpha = result.x
        # Should be equivalent to the sum of alpha*y_i*x_i
        
        # support vectors have alpha > 0
        self.__vecs_x = self.__x[self.__alpha > 1e-6]
        self.__vecs_y = self.__y[self.__alpha > 1e-6]
        self.__vecs_alpha = self.__alpha[self.__alpha > 1e-6]
        
        self.__w = np.sum((self.__alpha * self.__y)[:, np.newaxis] * self.__x, axis=0)
        
        # b is determined by y_i - w* x_i for some support vector (alpha > 0)
        # b = self.__vecs_y[0] - np.matmul(self.__vecs_x[0].T, self.__w)
        
        # in kernel mode we need to do the sum of all alpha*y*x[0]*x only of support vectors 
        b = self.__vecs_y[0] - np.sum( self.__vecs_alpha * self.__vecs_y * self.__kernel(self.__vecs_x[0], self.__vecs_x.T, **self.__kernel_kwargs))
        
        self.__w = np.array(list(self.__w) + [b])

    def __call__(self, data):
        # Previous call for regular SVM
        # return [ (1 if (sum(self.__w[i]*d[self.__col_idx[col]] for i,col in enumerate(self.__columns))+self.__w[-1])>=0 else -1) for d in data ]
        results = []
        for row in data:
            d = np.array([row[self.__col_idx[col]] for i,col in enumerate(self.__columns)])
            results.append(np.sum(self.__vecs_alpha * self.__vecs_y * self.__kernel(d, self.__vecs_x.T, **self.__kernel_kwargs)) + self.__w[-1])
            
        return [
            (1 if r>=0 else -1) for r in results
        ]
    
    
    def get_weights(self):
        return [w for w in self.__w]
    
    def get_support_vecs(self):
        return self.__vecs_x.copy()
