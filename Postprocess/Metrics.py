# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

def get_y(x, descriptor):
    target_idx = descriptor['columns'].index(descriptor['target'])
    return [d[target_idx] for d in x]

def accuracy(y, prediction):
    return sum( y[i]==prediction[i] for i in range(len(y)) )/len(y)

def square_error(y, prediction):
    return sum( (y[i]-prediction[i])**2 for i in range(len(y)) )/2
