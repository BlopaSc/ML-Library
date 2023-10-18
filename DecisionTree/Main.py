# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

import sys

from ID3 import ID3
sys.path.append("../Preprocess")
from CSVLoader import CSVLoader
from ReplaceMissingWithMajority import ReplaceMissingWithMajority
from DiscretizeNumericalAtMedian import DiscretizeNumericalAtMedian


# EDIT AS NECESSARY
car_train_path = 'car/train.csv'
car_test_path = 'car/test.csv'

bank_train_path = 'bank/train.csv'
bank_test_path = 'bank/test.csv'

car_descriptor = {
    'target': 'label',
    'columns': 'buying,maint,doors,persons,lug_boot,safety,label'.split(','),
    'categorical': 'buying,maint,doors,persons,lug_boot,safety,label'.split(',')
}

example_data = [['S','H','H','W','0'],['S','H','H','S','0'],['O','H','H','W','1'],['R','M','H','W','1'],['R','C','N','W','1'],['R','C','N','S','0'],['O','C','N','S','1'],['S','M','H','W','0'],['S','C','N','W','1'],['R','M','N','W','1'],['S','M','N','S','1'],['O','M','H','S','1'],['O','H','N','W','1'],['R','M','H','S','0']]
# example_data.append(['S','M','N','W','1'])
example_descriptor = {
    'target': 'play',
    'columns': 'O,T,H,W,play'.split(','),
    'categorical': 'O,T,H,W,play'.split(',')
}

bank_descriptor = {
    'target': 'y',
    'columns': 'age,job,marital,education,default,balance,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome,y'.split(','),
    'categorical': 'job,marital,education,default,housing,loan,contact,month,poutcome'.split(','),
    'numerical': 'age,balance,day,duration,campaign,pdays,previous'.split(',')
}


### MAIN
if __name__ == '__main__':
    # Section in for calculating the results of Part 2, question 2b
    space = ' & '
    end = ' \\\\'
    
    car_train = CSVLoader(car_train_path)
    car_test = CSVLoader(car_test_path)
    
    print("Car example")
    for i in range(1,6+1):
        res = []
        print(f"{i}", end='')
        for j in ['information_gain', 'gini_index', 'majority_error']:
            tree_car = ID3( car_train, car_descriptor, criterion = j, max_depth = i )
            _,acctrain =  tree_car.predict_and_error(car_train)
            _,acctest = tree_car.predict_and_error(car_test)
            print("{}{:.4f}{}{:.4f}".format(space, 1-sum(acctrain)/len(acctrain), space, 1-sum(acctest)/len(acctest)),end='')
        print(f"{end}")
        
    t  = ID3( example_data, example_descriptor, criterion='gini_index', max_depth=0)
    
    y = tree_car(car_test)
    
    print("Bank example with unknown as attribute value")
    bank_train = CSVLoader(bank_train_path,bank_descriptor)
    bank_test = CSVLoader(bank_test_path,bank_descriptor)
    
    discretizer = DiscretizeNumericalAtMedian(bank_train, bank_descriptor)
    
    bank_train = discretizer(bank_train, bank_descriptor)
    
    for i in range(1,16+1):
        res = []
        print(f"{i}", end='')
        for j in ['information_gain', 'gini_index', 'majority_error']:
            tree_bank = ID3( bank_train, bank_descriptor, criterion = j, max_depth = i, preprocess=[discretizer] )
            _,acctrain =  tree_bank.predict_and_error(bank_train)
            _,acctest = tree_bank.predict_and_error(bank_test)
            print("{}{:.4f}{}{:.4f}".format(space, 1-sum(acctrain)/len(acctrain), space, 1-sum(acctest)/len(acctest)),end='')
        print(f"{end}")
        
    replace = ReplaceMissingWithMajority(bank_train, bank_descriptor, missing = 'unknown')
    bank_train2 = replace(bank_train)
    
    print("Bank example without unknown as attribute value")
    for i in range(1,16+1):
        res = []
        print(f"{i}", end='')
        for j in ['information_gain', 'gini_index', 'majority_error']:
            tree_bank = ID3( bank_train2 , bank_descriptor, criterion = j, max_depth = i, preprocess=[discretizer,replace])
            _,acctrain =  tree_bank.predict_and_error(bank_train)
            _,acctest = tree_bank.predict_and_error(bank_test)
            print("{}{:.4f}{}{:.4f}".format(space, 1-sum(acctrain)/len(acctrain), space, 1-sum(acctest)/len(acctest)),end='')
        print(f"{end}")
        
        
        
        