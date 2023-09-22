# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

import math

class ID3:
    # Class initializer
        # data: Matrix-like structure which contains all the training data (labels included) used to design the tree
        # descriptor: a dictionary like structure which the followings keys:
            # target: name of the column which holds the target values
            # target_values: contains an array with the possible values for the target data, empty array if numerical
            # columns: contains an array with the name of each column in the data in their respective order
            # attributes: contains a dictionary in which each key is an attribute of the data
                # If the attribute is categorical then contains an array with its possible values
                # If the attribute is numerical then contains an empty array
        # criterion: Choose the criterion to use for the splitting, may be: 'information_gain'/'entropy' (default), 'gini_index' or 'majority_error'
        # max_depth: Maximum depth for the tree (counted as the longed number of edges from the root node to any other node), if negative it will go to the maximum possible depth
    def __init__(self, data, descriptor, criterion='entropy', max_depth=-1, missing=None):
        if criterion=='information_gain' or criterion=='entropy':
            self.criterion = self.__entropy
        elif criterion=='gini_index':
            self.criterion = self.__gini
        elif criterion=='majority_error':
            self.criterion = self.__me
        else:
            raise ValueError('Invalid value for criterion.')
        self.numeric =  {}
        self.majority = {}
        self.missing = missing
        for key in descriptor['attributes']:
            idx = descriptor['columns'].index(key)
            if missing and missing in descriptor['attributes'][key]:
                count = {v: sum(d[idx]==v for d in data) for v in descriptor['attributes'][key] if v!=missing}
                majority = max(zip(count.values(),count.keys()))[1]
                self.majority[key] = (idx,majority)
                for d in data:
                    if d[idx] == missing: d[idx] = majority
            if not descriptor['attributes'][key]:
                median = [d[idx] for d in data]
                median.sort()
                median = median[len(median)//2]
                for d in data:
                    d[idx] = '<=' if d[idx]<=median else '>'
                self.numeric[key] = (idx,median)
            
        self.columns = descriptor['columns']
        self.target_idx = self.columns.index(descriptor['target'])
        root = self.__build_leaf(None,data,descriptor)
        self.root = self.__split_node(root, descriptor, set([i for i in descriptor['attributes']]), max_depth)
    
    def __build_leaf(self, parent, data, descriptor):
        # Creates the basic structure of a node, even if this node will be discarded later
        # Stores: parent, count, label, n, data and criterion
        node = {'parent': parent, 'data': data, 'n': len(data), 'children': {}, 'attr': None}
        if data:
            if descriptor['target_values']:
                node['count'] = {key: sum(d[self.target_idx]==key for d in data) for key in descriptor['target_values']}
                node['label'] = max(zip(node['count'].values(),node['count'].keys()))[1]
            else:
                # TODO: How to label numerical?
                pass
        elif parent:
            node['label'] = parent['label']
        node['criterion'] = self.criterion(node)
        return node
    
    def __split_node(self, node, descriptor, attributes, max_depth):
        data = node['data']
        children = []
        if max_depth and data and node['criterion']!=0:
            # Split
            best_ig,best_attr = 0,None
            for attr in attributes:
                attr_idx = descriptor['columns'].index(attr)
                if descriptor['attributes'][attr]:
                    # Categorical
                    tmp_children = {v: self.__build_leaf(node, [d for d in data if d[attr_idx]==v], descriptor) for v in descriptor['attributes'][attr]}
                else:
                    # Numerical: Nothing, we turned them to binary in constructor
                    tmp_children = {v: self.__build_leaf(node, [d for d in data if d[attr_idx]==v], descriptor) for v in ['<=','>']}
                ig = node['criterion'] - sum((n['n']/node['n'])*n['criterion'] for n in tmp_children.values())
                if ig >= best_ig:
                    best_ig = ig
                    best_attr = attr
                    children = tmp_children
        if children:
            node['attr'] = best_attr
            node['children'] = {v: self.__split_node(children[v], descriptor, attributes - set([best_attr]), max_depth-1) for v in children}
        return node
    
    def __entropy(self, node):
        n = node['n']
        return -sum( ((v/n) * math.log2(v/n)) if v else 0 for v in node['count'].values() ) if n else 0
        
    def __gini(self, node):
        n = node['n']
        return (1 - sum((v/n)**2 for v in node['count'].values())) if n else 0
    
    def __me(self, node):
        n = node['n']
        return (1 - (node['count'][node['label']]/n)) if n else 0
    
    # x is a matrix-like structure of the items to predict, returns an array of labels
    def predict(self, x):
        pred = []
        for item in x:
            for key in self.majority:
                idx,majority = self.majority[key]
                if item[idx] == self.missing: item[idx] = majority
            for key in self.numeric:
                idx,median = self.numeric[key]
                if item[idx] != '<=' and item[idx] != '>':
                    item[idx] = '<=' if item[idx]<=median else '>'
            node = self.root
            while node['children']:
                node = node['children'][item[ self.columns.index(node['attr'])]]
            pred.append(node['label'])
        return pred
    
    # x is a matrix-like structure of the items to predict with their labels included, returns the array of predicted labels as well as the accuracy
    def predict_and_error(self, x):
        pred = []
        acc = 0
        for item in x:
            for key in self.majority:
                idx,majority = self.majority[key]
                if item[idx] == self.missing: item[idx] = majority
            for key in self.numeric:
                idx,median = self.numeric[key]
                if item[idx] != '<=' and item[idx] != '>':
                    item[idx] = '<=' if item[idx]<=median else '>'
            node = self.root
            while node['children']:
                node = node['children'][item[ self.columns.index(node['attr'])]]
            pred.append(node['label'])
            if node['label'] == item[self.target_idx]:
                acc += 1
        return (pred, 1 - (acc/len(x)))
            
        
