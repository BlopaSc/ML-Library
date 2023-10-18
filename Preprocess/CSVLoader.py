# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

def CSVLoader(path, descriptor={}, separator=','):
    data = []
    has_weight = 'weight' in descriptor
    weight_idx = descriptor['columns'].index(descriptor['weight']) if has_weight else None
    with open(path, 'r') as f:
        for line in f:
            data.append([i for i in line.strip().split(separator) if i])
    numerical_idx = [descriptor.get('columns','').index(col) for col in descriptor.get('numerical',[])]
    if numerical_idx:
        for d in data:
            for idx in numerical_idx:
                d[idx] = float(d[idx])
            if has_weight: d[weight_idx] = float(d[weight_idx])
    return data

