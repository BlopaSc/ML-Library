# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

import math

# TODO: Preprocessing convert numerical to categorical, fill nulls with desired technique, calculate weights

class ID3:
    # Descriptor includes:
        # 'target': name of column for target, strictly categorical
        # 'columns': list of columns in their respective order
        # 'categorical': set of names of categorical columns to use as attributes
        # 'numerical': set of names of numerical columns to use as attributes
        # 'weight': (optional) name of the column used to weight the data
    
    def __init__(self, data, descriptor, criterion='entropy', max_depth=0, preprocess=[]):
        # Store parameters
        if criterion=='information_gain' or criterion=='entropy':
            self.__criterion = self.__entropy
        elif criterion=='gini_index' or criterion=='gini':
            self.__criterion = self.__gini
        elif criterion=='majority_error' or criterion=='majority':
            self.__criterion = self.__me
        self.__max_depth = max_depth
        # Quick access to columns
        self.__columns = descriptor['columns']
        self.__col_idx = { col : self.__columns.index(col) for col in self.__columns }
        # Quick access to weight and specify weight function
        self.__weight = self.__sum if descriptor.get('weight') else self.__count
        self.__weight_idx = self.__columns.index(descriptor.get('weight')) if descriptor.get('weight') else None
        # Convert to sets if needed
        self.__categorical = set(descriptor.get('categorical',[]))
        self.__numerical = set(descriptor.get('numerical',[]))
        # if not self.__categorical: self.__categorical = set(self.__columns) - (self.__numerical | set([descriptor.get('weight','')]) )
        # if not self.__numerical: self.__numerical = set(self.__columns) - (self.__categorical | set([descriptor.get('weight','')]) )
        # Quick access and set of values of the target
        self.__target_idx = self.__col_idx[descriptor['target']]
        self.__vtarget = set([d[self.__target_idx] for d in data])
        self.__vattr = { attr: set([d[self.__col_idx[attr]] for d in data]) for attr in descriptor.get('categorical',[]) }
        # Copies self data, calls preprocessors' constructors, applies to data
        self.__preprocessors = [p for p in preprocess]
        # Construct tree
        root = self.__build_leaf(None,data)
        self.__root = self.__split_node(root, (self.__categorical | self.__numerical) - set([descriptor['target']]))
    
    def __repr__(self):
        return self.__print_node(self.__root)
    
    def __print_node(self,node):
        result = ' '*(node['depth']*2) + ' '.join([str(node[d]) for d in ['n','label','criterion','attr']]) + '\n'
        for k in node['children']:
            result += self.__print_node(node['children'][k])
        return result
    
    # Calls the preprocessors sequentially on a single sample or matrix-like structure and returns the corresponding sample or samples
    def __preprocess(self,data):
        r = data if type(data[0]) == list else [data]
        for p in self.__preprocessors:
            r = p(r)
        return r
    
    # Weight is number of elements in data
    def __count(self, data):
        return len(data)
    
    # Weight is sum of weight values in data
    def __sum(self,data):
        return sum(d[self.__weight_idx] for d in data)
    
    # Builds a leaf node
    def __build_leaf(self, parent, data):
        # Stores: data, weight, depth
            # Stubs for children and attr
        node = {'data': data, 'n': self.__weight(data), 'children': {}, 'attr': None, 'depth': parent['depth']+1 if parent else 0}
        if data:
            node['count'] = {key: self.__weight([d for d in data if d[self.__target_idx]==key]) for key in self.__vtarget}
            node['label'] = max(zip(node['count'].values(),node['count'].keys()))[1]
        elif parent:
            node['label'] = parent['label']
        node['criterion'] = self.__criterion(node)
        return node
    
    def __split_node(self, node, attributes):
        data = node['data']
        children = []
        if (node['depth']<self.__max_depth or self.__max_depth<=0) and data and node['criterion']!=0:
            # Split
            best_ig,best_attr,best_value = 0,None,None
            for attr in attributes:
                attr_idx = self.__col_idx[attr]
                if attr in self.__categorical:
                    # Categorical
                    tmp_children = {v: self.__build_leaf(node, [d for d in data if d[attr_idx]==v]) for v in self.__vattr[attr]}
                elif attr in self.__numerical:
                    # Numerical
                    if self.__weight_idx:
                        tmp_value = sum(d[attr_idx]*d[self.__weight_idx] for d in data)/node['n']
                    else:
                        tmp_value = sum(d[attr_idx] for d in data)/node['n']
                    tmp_children = {'<=': self.__build_leaf(node, [d for d in data if d[attr_idx]<=tmp_value]),
                                    '>': self.__build_leaf(node, [d for d in data if d[attr_idx]>tmp_value]) }
                ig = node['criterion'] - sum((n['n']/node['n'])*n['criterion'] for n in tmp_children.values())
                if ig >= best_ig:
                    best_ig = ig
                    best_attr = attr
                    if attr in self.__numerical: best_value = tmp_value
                    children = tmp_children
        if children:
            node['attr'] = best_attr
            node['children'] = {v: self.__split_node(children[v], attributes - set([best_attr])) for v in children}
            if best_attr in self.__numerical:
                node['value'] = best_value
        return node
    
    def __entropy(self, node):
        n = node['n']
        return -sum( (((v/n) * math.log2(v/n)) if v else 0) for v in node['count'].values() ) if n else 0
        
    def __gini(self, node):
        n = node['n']
        return (1 - sum((v/n)**2 for v in node['count'].values())) if n else 0
    
    def __me(self, node):
        n = node['n']
        return (1 - (node['count'][node['label']]/n)) if n else 0
    
    # x is a matrix-like structure of the items to predict, returns an array of labels
    def __call__(self, x):
        pred = []
        for item in x:
            it = self.__preprocess(item)
            value = {v: 0 for v in self.__vtarget}
            for i in it:
                node = self.__root
                while node['children']:
                    if node['attr'] in self.__categorical:
                        nnode = node['children'].get(i[ self.__col_idx[node['attr']]])
                        if nnode:
                            node = nnode
                        else:
                            break
                    elif node['attr'] in self.__numerical:
                        node = node['children']['<='] if (float(i[self.__col_idx[node['attr']]]) <= node['value']) else node['children']['>']
                    else:
                        print("Cat:", self.__categorical)
                        print("Num:", self.__numerical)
                        print(f'UNKNOWN ATTRIBUTE: "{node["attr"]}"')
                # Note: weighted results should be handed by the user
                # value[node['label']] += self.__weight( [i] )
                value[node['label']] += 1
            label = max(zip(value.values(),value.keys()))[1]
            pred.append(label)
        return pred
    
    # x is a matrix-like structure of the items to predict but also includes labels, returns an array of labels and an array of true/false depending on whether the prediction was correct
    def predict_and_error(self,x):
        pred = self(x)
        correct = [p==y[self.__target_idx] for p,y in zip(pred,x)]
        return (pred,correct)