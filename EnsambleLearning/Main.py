# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

import sys

from AdaBoost import AdaBoost
from Bagging import Bagging
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
        
    
    
    
    