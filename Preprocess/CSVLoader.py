# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

import copy

def CSVLoader(path, descriptor={}, separator=',',skip=0):
    data = []
    has_weight = 'weight' in descriptor
    weight_idx = descriptor['columns'].index(descriptor['weight']) if has_weight else None
    with open(path, 'r') as f:
        for line in f:
            data.append([i for i in line.strip().split(separator) if i])
    for i in range(skip): data.pop(0)
    numerical_idx = [descriptor.get('columns','').index(col) for col in descriptor.get('numerical',[])]
    if numerical_idx:
        for d in data:
            for idx in numerical_idx:
                d[idx] = float(d[idx])
            if has_weight: d[weight_idx] = float(d[weight_idx])
    return data

def remove_column(data, descriptor, column):
    column_idx = descriptor['columns'].index(column)
    new_data = [[d for i,d in enumerate(row) if i!=column_idx] for row in data]
    new_descriptor = copy.deepcopy(descriptor)
    new_descriptor['columns'].pop(column_idx)
    if column in new_descriptor['categorical']:
        new_descriptor['categorical'] = set(new_descriptor['categorical']) - set([column])
    if column in new_descriptor['numerical']:
        new_descriptor['numerical'] = set(new_descriptor['numerical']) - set([column])
    return new_data,new_descriptor

def remove_columns(data, descriptor, columns):
    for col in columns:
        data,descriptor = remove_column(data, descriptor, col)
    return data,descriptor

def add_column(data, descriptor, column_data, column_name, typ='categorical'):
    new_data = [d + [column_data[i]] for i,d in enumerate(data)]
    new_descriptor = copy.deepcopy(descriptor)
    new_descriptor['columns'].append(column_name)
    new_descriptor[typ] = set(new_descriptor[typ]) | set([column_name])
    return new_data,new_descriptor

def get_column(data, descriptor, column):
    column_idx = descriptor['columns'].index(column)
    return [d[column_idx] for d in data]
    
def swap_columns(data, descriptor, column1, column2):
    new_data = [[i for i in d] for d in data]
    new_descriptor = copy.deepcopy(descriptor)
    col1_idx = descriptor['columns'].index(column1)
    col2_idx = descriptor['columns'].index(column2)
    for d in new_data:
        tmp = d[col1_idx]
        d[col1_idx] = d[col2_idx]
        d[col2_idx] = tmp
    tmp = new_descriptor['columns'][col1_idx]
    new_descriptor['columns'][col1_idx] = new_descriptor['columns'][col2_idx]
    new_descriptor['columns'][col2_idx] = tmp
    return new_data,new_descriptor

def replace_value(data, descriptor, column, value, newvalue):
    column_idx = descriptor['columns'].index(column)
    return [[(newvalue if (i==column_idx and d==value) else d) for i,d in enumerate(row)] for row in data]
    