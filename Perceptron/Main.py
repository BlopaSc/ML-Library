# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

import sys

from Perceptron import Perceptron
from VotedPerceptron import VotedPerceptron
from AveragePerceptron import AveragePerceptron
sys.path.append("../Preprocess")
from CSVLoader import CSVLoader,replace_value
sys.path.append("../Postprocess")
from Metrics import *


try:
    import matplotlib.pyplot as plt
    plot = True
except ImportError:
    print("Missing matplotlib.pyplot")
    plot = False

banknote_train_path = 'bank-note/train.csv'
banknote_test_path = 'bank-note/test.csv'
banknote_descriptor = {
    'target': 'genuine',
    'columns': 'variance,skewness,curtosis,entropy,genuine'.split(','),
    'numerical': 'variance,skewness,curtosis,entropy,genuine'.split(',')
}

def error(y, pred):
    return 1 - accuracy(y,pred)

def test_model(model, T, step, x_train, x_test, y_train, y_test, error_function):
    train_errors = []
    test_errors = []
    for i in range(1,T+1,step):
        model.modify_epochs(i)
        train_pred = model(x_train)
        test_pred = model(x_test)
        train_errors.append(error_function(y_train,train_pred))
        test_errors.append(error_function(y_test,test_pred))
        print("T=",i,train_errors[-1],test_errors[-1])
    if plot:
        x = [i for i in range(1,T+1,step)]
        plt.figure(figsize=(10,6))
        plt.plot(x,train_errors, 'b')
        plt.plot(x,test_errors, 'r')
        plt.legend(["Train","Test"])
        plt.xlabel("Iterations")
        plt.ylabel("Error")
        plt.show()

def beauty_weight(w, separator=', '):
    return separator.join(f'{i:.4f}' for i in w)

### MAIN
if __name__ == '__main__':
    banknote_train = CSVLoader(banknote_train_path, banknote_descriptor)
    banknote_test = CSVLoader(banknote_test_path, banknote_descriptor)
    
    banknote_train = replace_value(banknote_train, banknote_descriptor, 'genuine', 0, -1.0)
    banknote_test = replace_value(banknote_test, banknote_descriptor, 'genuine', 0, -1.0)
    
    y_train = get_y(banknote_train, banknote_descriptor)
    y_test = get_y(banknote_test, banknote_descriptor)

    T = 10
    
    print("Normal Perceptron:")
    perceptron = Perceptron(banknote_train, banknote_descriptor, max_iters = 0, seed=2048)
    test_model(perceptron, T, 1, banknote_train, banknote_test, y_train, y_test, error)
    print("Final w:", beauty_weight(perceptron.get_weights()))
    
    print("Voted Perceptron:")
    voted = VotedPerceptron(banknote_train, banknote_descriptor, max_iters = 0)
    test_model(voted, T, 1, banknote_train, banknote_test, y_train, y_test, error)
    print("Weights")
    total = [0 for i in range(5)]
    for c,w in voted.get_weights():
        total = [total[i] + c*w[i] for i in range(5)]
        print(str(c)+' & ('+beauty_weight(w) + ') \\\\')
        
    print("Average Perceptron:")
    average = AveragePerceptron(banknote_train, banknote_descriptor, max_iters = 0)
    test_model(average, T, 1, banknote_train, banknote_test, y_train, y_test, error)
    print("Final w:", beauty_weight(average.get_weights()))
    
    print("Now compare the total of counter*weight of the voted perceptron with the final w of the average perceptron:")
    print("Averg", beauty_weight(total))
    print("Voted", beauty_weight(average.get_weights()))
    
    
