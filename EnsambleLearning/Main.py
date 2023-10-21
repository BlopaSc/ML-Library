# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

import random
import sys

from AdaBoost import AdaBoost
from Bagging import Bagging
from RandomForest import RandomForest
sys.path.append("../Preprocess")
from CSVLoader import CSVLoader
from DiscretizeNumericalAtMedian import DiscretizeNumericalAtMedian
sys.path.append("../Postprocess")
from Metrics import *

try:
    import matplotlib.pyplot as plt
    plot = True
except ImportError:
    print("Missing matplotlib.pyplot")
    plot = False

bank_train_path = 'bank/train.csv'
bank_test_path = 'bank/test.csv'
bank_descriptor = {
    'target': 'y',
    'columns': 'age,job,marital,education,default,balance,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome,y'.split(','),
    'categorical': 'job,marital,education,default,housing,loan,contact,month,poutcome'.split(','),
    'numerical': 'age,balance,day,duration,campaign,pdays,previous'.split(',')
}

example_data = [['S','H','H','W','0',1],['S','H','H','S','0',1],['O','H','H','W','1',1],['R','M','H','W','1',1],['R','C','N','W','1',1],['R','C','N','S','0',1],['O','C','N','S','1',1],['S','M','H','W','0',1],['S','C','N','W','1',1],['R','M','N','W','1',1],['S','M','N','S','1',1],['O','M','H','S','1',1],['O','H','N','W','1',1],['R','M','H','S','0',1]]
example_descriptor = {
    'target': 'play',
    'columns': 'O,T,H,W,play,w'.split(','),
    'categorical': 'O,T,H,W,play'.split(','),
    'weight': 'w'
}

T = 500

### MAIN
if __name__ == '__main__':
    bank_train = CSVLoader(bank_train_path,bank_descriptor)
    bank_test = CSVLoader(bank_test_path,bank_descriptor)
    
    discretizer = DiscretizeNumericalAtMedian(bank_train, bank_descriptor)
    
    bank_train = discretizer(bank_train, bank_descriptor)
    bank_test = discretizer(bank_test)
    
    train_y = get_y(bank_train, bank_descriptor)
    test_y = get_y(bank_test, bank_descriptor)
    
    print("Adaboost exercise errors")
    ada = AdaBoost(bank_train, bank_descriptor, 0, max_depth=1)
    ada_train_errors = []
    ada_test_errors = []
    ada_train_errors_last_stump = []
    ada_test_errors_last_stump = []
    for i in range(1,T+1):
        ada.modify_T(i)
        train_pred = ada(bank_train)
        test_pred = ada(bank_test)
        ada_train_errors.append(1-accuracy(train_y, train_pred))
        ada_test_errors.append(1-accuracy(test_y, test_pred))
        train_pred_last_stump = ada.predict_with_stump(bank_train, i-1)
        test_pred_last_stump = ada.predict_with_stump(bank_test, i-1)
        ada_train_errors_last_stump.append(1-accuracy(train_y, train_pred_last_stump))
        ada_test_errors_last_stump.append(1-accuracy(test_y, test_pred_last_stump))
        
        print("T=",i,ada_train_errors[-1],ada_test_errors[-1])
    
    if plot:
        plt.figure(figsize=(10,6))
        plt.plot(ada_train_errors, 'b')
        plt.plot(ada_test_errors, 'r')
        plt.legend(["Train","Test"])
        plt.xlabel("Number of classifiers")
        plt.ylabel("Error")
        plt.show()
    if plot:
        plt.figure(figsize=(10,6))
        plt.plot(ada_train_errors_last_stump, 'b')
        plt.plot(ada_test_errors_last_stump, 'r')
        plt.legend(["Train","Test"])
        plt.xlabel("Number of classifier")
        plt.ylabel("Error")
        plt.show()
    
    print("Bagging exercise errors")
    bag_train_errors = []
    bag_test_errors = []
    bag = Bagging(bank_train, bank_descriptor, 0, m=2000, seed=1337)
    for i in range(1,T+1):
        bag.modify_T(i)
        train_pred = bag(bank_train)
        test_pred = bag(bank_test)
        bag_train_errors.append(1-accuracy(train_y, train_pred))
        bag_test_errors.append(1-accuracy(test_y, test_pred))
        print("T=",i,bag_train_errors[-1],bag_test_errors[-1])
        
    if plot:
        plt.figure(figsize=(10,6))
        plt.plot(bag_train_errors, 'b')
        plt.plot(bag_test_errors, 'r')
        plt.legend(["Train","Test"])
        plt.xlabel("Number of classifiers")
        plt.ylabel("Error")
        plt.show()
        
    def get_bias_variance(predictions, test_y, name):
        bias_sample = [
            ( (sum((1 if pred[i]=='yes' else 0) for pred in predictions)/len(predictions)) - (1 if test_y[i]=='yes' else 0))**2
            for i in range(len(test_y))
        ]
        
        variance = [
            sum((bias_sample[i] - (1 if pred[i]=='yes' else 0))**2 for i in range(len(bias_sample)))/len(bias_sample)
            for pred in predictions
        ]
        print(name,"average accuracy:", sum(accuracy(test_y, pred) for pred in predictions)/len(predictions))
        print(name,"average bias:",sum(bias_sample)/len(bias_sample))
        print(name,"average variance:",sum(variance)/len(variance) )
        
    print("Bias and Variance exercise")
    bags = []
    prng = random.Random()
    prng.seed(21)
    for i in range(100):
        subdata = prng.choices(bank_train, k=1000)
        bags.append( Bagging(subdata, bank_descriptor, 500, m=500, seed=21+i) )
    
    single_tree_preds = []
    for b in bags:
        single_tree_preds.append(b.predict_with_tree(bank_test, 0))
    get_bias_variance(single_tree_preds, test_y, "Single tree")
    
    ensamble_tree_pred = []
    for b in bags:
        ensamble_tree_pred.append(b(bank_test))
    get_bias_variance(ensamble_tree_pred, test_y, "Ensamble")
    
    print("Random forest exercise errors")
    forest2_train_errors = []
    forest2_test_errors = []
    forest4_train_errors = []
    forest4_test_errors = []
    forest6_train_errors = []
    forest6_test_errors = []
    forest2 = RandomForest(bank_train, bank_descriptor, 0, 2, m=2000, seed=1337)
    forest4 = RandomForest(bank_train, bank_descriptor, 0, 4, m=2000, seed=1337)
    forest6 = RandomForest(bank_train, bank_descriptor, 0, 6, m=2000, seed=1337)
    for i in range(1,T+1):
        forest2.modify_T(i)
        forest4.modify_T(i)
        forest6.modify_T(i)
        train_pred = forest2(bank_train)
        test_pred = forest2(bank_test)
        forest2_train_errors.append(1-accuracy(train_y, train_pred))
        forest2_test_errors.append(1-accuracy(test_y, test_pred))
        
        train_pred = forest4(bank_train)
        test_pred = forest4(bank_test)
        forest4_train_errors.append(1-accuracy(train_y, train_pred))
        forest4_test_errors.append(1-accuracy(test_y, test_pred))
        
        train_pred = forest6(bank_train)
        test_pred = forest6(bank_test)
        forest6_train_errors.append(1-accuracy(train_y, train_pred))
        forest6_test_errors.append(1-accuracy(test_y, test_pred))
        
        print("T=",i,forest2_train_errors[-1],forest2_test_errors[-1],forest4_train_errors[-1],forest4_test_errors[-1],forest6_train_errors[-1],forest6_test_errors[-1])
        
    if plot:
        plt.figure(figsize=(10,6))
        plt.plot(forest2_train_errors, 'b')
        plt.plot(forest4_train_errors, 'c')
        plt.plot(forest6_train_errors, 'g')
        plt.plot(forest2_test_errors, 'r')
        plt.plot(forest4_test_errors, 'm')
        plt.plot(forest6_test_errors, '#FFA500')
        plt.legend(["Train - subsample 2","Train - subsample 4","Train - subsample 6","Test - subsample 2", "Test - subsample 4", "Test - subsample 6"])
        plt.xlabel("Number of classifiers")
        plt.ylabel("Error")
        plt.show()
    
    print("Bias and Variance exercise for forests")
    forests = []
    prng = random.Random()
    prng.seed(21)
    for i in range(100):
        subdata = prng.choices(bank_train, k=1000)
        forests.append( RandomForest(subdata, bank_descriptor, 500, 4, m=500, seed=21+i) )
    
    single_tree_preds = []
    for f in forests:
        single_tree_preds.append(f.predict_with_tree(bank_test, 0))
    get_bias_variance(single_tree_preds, test_y, "Single tree")
    
    ensamble_tree_pred = []
    for f in forests:
        ensamble_tree_pred.append(f(bank_test))
    get_bias_variance(ensamble_tree_pred, test_y, "Ensamble forest")
    
    