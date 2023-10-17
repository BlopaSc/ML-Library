# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

class DiscretizeNumericalAtMedian:
    # Discritizes a selection of numerical columns into a binary by their median: <= and >
    def __init__(self, data, descriptor, columns=[]):
        if not columns: columns = set(descriptor['columns'])
        self.__columns = [(col,descriptor['columns'].index(col)) for col in columns if col in descriptor['numerical']]
        self.__medians = {}
        if 'weight' in descriptor:
            weight_idx = descriptor['columns'].index(descriptor['weight'])
            n = sum(d[weight_idx] for d in data)
            median,tot = 0,0
            while tot < n/2:
                tot += data[median][weight_idx]
                if tot < n/2: median += 1
        else:
            median = len(data)//2
        for col,idx in self.__columns:
            values = [d[idx] for d in data]
            values.sort()
            self.__medians[col] = values[median]
    
    # If called with descriptor, modifies descriptor
    def __call__(self, data, descriptor={}):
        result = []
        for d in data:
            item = [item for item in d]
            for col,idx in self.__columns:
                if type(item[idx]) == float: item[idx] = '<=' if item[idx]<=self.__medians[col] else '>'
            result.append(item)
        if descriptor:
            descriptor['numerical'] = set(descriptor.get('numerical',[])) - set([col for col,_ in self.__columns])
            descriptor['categorical'] = set(descriptor.get('categorical',[])) | set([col for col,_ in self.__columns])
        return result
        
