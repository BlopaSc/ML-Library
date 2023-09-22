# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

from ID3 import ID3

# EDIT AS NECESSARY
car_train_path = 'car/train.csv'
car_test_path = 'car/test.csv'

bank_train_path = 'bank/train.csv'
bank_test_path = 'bank/test.csv'

def load_csv(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append([i for i in line.strip().split(',') if i])
    return data

car_descriptor = {
    'target': 'label',
    'target_values': 'unacc,acc,good,vgood'.split(','),
    'attributes': {
        'buying': 'vhigh,high,med,low'.split(','),
        'maint': 'vhigh,high,med,low'.split(','),
        'doors': '2,3,4,5more'.split(','),
        'persons':  '2,4,more'.split(','),
        'lug_boot': 'small,med,big'.split(','),
        'safety': 'low,med,high'.split(',')
    },
    'columns': 'buying,maint,doors,persons,lug_boot,safety,label'.split(',')
}


example_data = [['S','H','H','W','0'],['S','H','H','S','0'],['O','H','H','W','1'],['R','M','H','W','1'],['R','C','N','W','1'],['R','C','N','S','0'],['O','C','N','S','1'],['S','M','H','W','0'],['S','C','N','W','1'],['R','M','N','W','1'],['S','M','N','S','1'],['O','M','H','S','1'],['O','H','N','W','1'],['R','M','H','S','0']]
example_data.append(['S','M','N','W','1'])
example_descriptor = {
    'target': 'play',
    'target_values': '0,1'.split(','),
    'attributes': {
        'O': 'S,O,R'.split(','),
        'T': 'C,M,H'.split(','),
        'H': 'L,N,H'.split(','),
        'W':  'W,S'.split(','),
    },
    'columns': 'O,T,H,W,play'.split(',')
}

bank_descriptor = {
    'target': 'y',
    'target_values': ["yes","no"],
    'attributes': {
        'age': [],
        'job': ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"],
        'marital': ["married","divorced","single"],
        'education': ["unknown","secondary","primary","tertiary"],
        'default': ["yes","no"],
        'balance': [],
        'housing': ["yes","no"],
        'loan': ["yes","no"],
        'contact': ["unknown","telephone","cellular"],
        'day': [],
        'month': ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        'duration': [],
        'campaign': [],
        'pdays': [],
        'previous': [],
        'poutcome': ["unknown","other","failure","success"]
    },
    'columns': 'age,job,marital,education,default,balance,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome,y'.split(',')
}

### MAIN
if __name__ == '__main__':
    # Section in for calculating the results of Part 2, question 2b
    space = ' & '
    end = ' \\\\'
    
    print("Car example")
    for i in range(1,6+1):
        res = []
        print(f"{i}", end='')
        for j in ['information_gain', 'gini_index', 'majority_error']:
            car_train = load_csv(car_train_path)
            car_test = load_csv(car_test_path)
            tree_car = ID3( car_train, car_descriptor, criterion = j, max_depth = i )
            _,acctrain =  tree_car.predict_and_error(car_train)
            _,acctest = tree_car.predict_and_error(car_test)
            print("{}{:.4f}{}{:.4f}".format(space, acctrain, space, acctest),end='')
        print(f"{end}")
    
    class_example = ID3(example_data, example_descriptor, criterion='entropy')
    
    print("Bank example with unknown as attribute value")
    for i in range(1,16+1):
        res = []
        print(f"{i}", end='')
        for j in ['information_gain', 'gini_index', 'majority_error']:
            bank_train = load_csv(bank_train_path)
            bank_test = load_csv(bank_test_path)
            tree_bank = ID3( bank_train, bank_descriptor, criterion = j, max_depth = i )
            _,acctrain =  tree_bank.predict_and_error(bank_train)
            _,acctest = tree_bank.predict_and_error(bank_test)
            print("{}{:.4f}{}{:.4f}".format(space, acctrain, space, acctest),end='')
        print(f"{end}")
    
    print("Bank example without unknown as attribute value")
    for i in range(1,16+1):
        res = []
        print(f"{i}", end='')
        for j in ['information_gain', 'gini_index', 'majority_error']:
            bank_trainm = load_csv(bank_train_path)
            bank_testm = load_csv(bank_test_path)
            tree_bank = ID3( bank_trainm, bank_descriptor, criterion = j, max_depth = i, missing='unknown' )
            _,acctrain =  tree_bank.predict_and_error(bank_trainm)
            _,acctest = tree_bank.predict_and_error(bank_testm)
            print("{}{:.4f}{}{:.4f}".format(space, acctrain, space, acctest),end='')
        print(f"{end}")