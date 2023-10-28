# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

import math

class DiscretizeNumericalWithNormal:
    # Discritizes a selection of numerical columns into a binary by their median: <= and >
    def __init__(self, data, descriptor, columns=[]):
        if not columns: columns = set(descriptor['columns'])
        self.__columns = [(col,descriptor['columns'].index(col)) for col in columns if col in descriptor['numerical']]
        self.__averages = {col: 0 for col,idx in self.__columns}
        self.__stds = {col: 0 for col,idx in self.__columns}
        weight_exists = 'weight' in descriptor
        weight_idx = descriptor['columns'].index(descriptor['weight']) if 'weight' in descriptor else None
        n = 0
        for d in data:
            n += d[weight_idx] if weight_exists else 1
            for col,idx in self.__columns:
                self.__averages[col] += d[idx]
        for col,_ in self.__columns:
            self.__averages[col] /= n
        for d in data:
            for col,idx in self.__columns:
                self.__stds[col] += (d[idx] - self.__averages[col])**2
        for col,_ in self.__columns:
            self.__stds[col] = math.sqrt(self.__stds[col]/n)

    # If called with descriptor, modifies descriptor
    def __call__(self, data, descriptor={}):
        result = []
        for d in data:
            item = [item for item in d]
            for col,idx in self.__columns:
                if type(item[idx]) == float: 
                    item[idx] = str(round((item[idx]-self.__averages[col])/self.__stds[col]))
            result.append(item)
        if descriptor:
            descriptor['numerical'] = set(descriptor.get('numerical',[])) - set([col for col,_ in self.__columns])
            descriptor['categorical'] = set(descriptor.get('categorical',[])) | set([col for col,_ in self.__columns])
        return result
        
