# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

import sys

from AdaBoost import AdaBoost
sys.path.append("../Preprocess")
from CSVLoader import CSVLoader
from DiscretizeNumericalAtMedian import DiscretizeNumericalAtMedian
sys.path.append("../Postprocess")
from Metrics import *

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

### MAIN
if __name__ == '__main__':
    bank_train = CSVLoader(bank_train_path,bank_descriptor)
    bank_test = CSVLoader(bank_test_path,bank_descriptor)
    
    discretizer = DiscretizeNumericalAtMedian(bank_train, bank_descriptor)
    
    bank_train = discretizer(bank_train, bank_descriptor)
    bank_test = discretizer(bank_test)
    
    train_y = get_y(bank_train, bank_descriptor)
    test_y = get_y(bank_test, bank_descriptor)
    
    ada = AdaBoost(bank_train, bank_descriptor, 0, max_depth=1)
    
    for i in range(1,10+1):
        ada.modify_T(i)
        
        train_pred = ada(bank_train)
        test_pred = ada(bank_test)
        
        print(accuracy(train_y, train_pred), accuracy(test_y, test_pred))
        
        
    
    
    