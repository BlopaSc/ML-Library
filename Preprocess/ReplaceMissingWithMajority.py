# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

class ReplaceMissingWithMajority:
    # Replaces all missing values from the columns listed in columns with the majority which shares the same label
    # If columns is empty then it will check all columns for the missing value
    def __init__(self, data, descriptor, missing, columns=[]):
        if type(columns) == str: columns = [columns]
        if not columns: columns = set(descriptor['columns']) - set(descriptor['target'])
        self.__columns = [(col,descriptor['columns'].index(col)) for col in columns if col in descriptor['categorical']]
        self.__missing = missing
        self.__counts_total = {col: {} for col,_ in self.__columns}
        self.__counts_label = {col: {} for col,_ in self.__columns}
        self.__target_idx = descriptor['columns'].index(descriptor['target'])
        # Counts appearance of attribute per label in each column
        for d in data:
            for col,idx in self.__columns:
                if d[idx] != missing:
                    self.__counts_total[col][d[idx]] = self.__counts_total[col].get(d[idx],0) + 1 
                    if not d[self.__target_idx] in self.__counts_label[col]: self.__counts_label[col][d[self.__target_idx]] = {}
                    self.__counts_label[col][d[self.__target_idx]][d[idx]] = self.__counts_label[col][d[self.__target_idx]].get(d[idx],0) + 1
        self.__majority_total ={col: {} for col,_ in self.__columns}
        self.__majority_label ={col: {} for col,_ in self.__columns}
        for col,_ in self.__columns:
            for key in self.__counts_label[col]:
                self.__majority_label[col][key] = max(zip(self.__counts_label[col][key].values(),self.__counts_label[col][key].keys()))[1]
            self.__majority_total[col] = max(zip(self.__counts_total[col].values(),self.__counts_total[col].keys()))[1]
    
    # Executes the preprocessing algorithm corresponding to either the train or test data
    def __call__(self, data, labeled):
        result = []
        if labeled:
            for d in data:
                nd = [item for item in d]
                # print(d)
                # print(nd)
                for col,idx in self.__columns:
                    # print(col,idx)
                    if nd[idx] == self.__missing:
                        nd[idx] = self.__majority_label[col][nd[self.__target_idx]]
                result.append(nd)
        else:
            for d in data:
                nd = [item for item in d]
                for col,idx in self.__columns:
                    if nd[idx] == self.__missing:
                        nd[idx] = self.__majority_total[col]
                result.append(nd)
        return result
