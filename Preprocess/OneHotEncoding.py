# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

from CSVLoader import remove_column,add_column,get_column

def OneHotEncoding(data, descriptor, columns=[]):
    if not columns: columns = descriptor['categorical']
    for col in columns:
        col_idx = descriptor['columns'].index(col)
        unique = set([d[col_idx] for d in data])
        new_columns = {key:[0 for i in range(len(data))] for key in unique}
        for i,d in enumerate(data):
            new_columns[d[col_idx]][i] = 1
        data,descriptor = remove_column(data, descriptor, col)
        for key in new_columns:
            data,descriptor = add_column(data,descriptor,new_columns[key],col+'_'+key)
    target = descriptor.get('target')
    if target:
        y = get_column(data, descriptor, target)
        data,descriptor = remove_column(data, descriptor, target)
        data,descriptor = add_column(data, descriptor, y, target)
    return data,descriptor
