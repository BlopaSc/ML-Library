# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

import numpy as np
import sys

from NeuralNetwork import NeuralNetwork

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

def test_model(model, T, step, x_train, x_test, y_train, y_test, error_function, do_plot=True, title='', print_step=1):
    train_errors = []
    test_errors = []
    for i in range(1,T+1,step):
        model.modify_epochs(i)
        train_pred = np.round(model(x_train))
        test_pred = np.round(model(x_test))
        train_errors.append(error_function(y_train,train_pred))
        test_errors.append(error_function(y_test,test_pred))
        if (i%print_step)==0: print("T=",i,train_errors[-1],test_errors[-1])
    if plot and do_plot:
        x = [i for i in range(1,T+1,step)]
        plt.figure(figsize=(10,6))
        plt.plot(x,train_errors, 'b')
        plt.plot(x,test_errors, 'r')
        plt.legend(["Train","Test"])
        plt.xlabel("Iterations")
        plt.ylabel("Error")
        if title: plt.title(title)
        plt.show()
    return train_errors,test_errors

### MAIN
if __name__ == '__main__':
    
    # Paper exercises
    print("Results for the NN of the paper problems")
    weights = [[[-2,2], [-3,3], [-1,1]] , [[-2,2], [-3,3], [-1,1]], [[2], [-1.5], [-1]]]
    nn_practice = NeuralNetwork([], {'target': 'y', 'columns': 'x1,x2,y'.split(',')} ,layers=[2,2,2,1], function='ssn', weights=weights)
    print("Result of forward propagation for [1,1,1]:", nn_practice.forward(np.array([[1,1]]).T))
    nn_practice.backward(np.array([[1]]))
    print("Result of backward propagation for [1,1,1] with y=1")
    for i,dw in enumerate(nn_practice.get_gradients()):
        print("Gradients for layer",i+1)
        print(dw)
    print("Note: bias are appended as a last value (x_n) instead of a first (x_0), therefore the values for bias show last")
    
    
    print("\nResults for the practice problems")
    banknote_train = CSVLoader(banknote_train_path, banknote_descriptor)
    banknote_test = CSVLoader(banknote_test_path, banknote_descriptor)
    
    y_train = get_y(banknote_train, banknote_descriptor)
    y_test = get_y(banknote_test, banknote_descriptor)

    NEURON_WIDTHS = [5, 10, 25, 50, 100]
    functions = 'sss'

    T = 100
    
    print("Evaluating NN with varying neurons in hidden layer")
    # results = {}
    # for N in NEURON_WIDTHS:
    #     print("Tuning for",N,"neurons")
    #     for i in range(2,15,1):
    #         for j in range(max(0,i-3),i+3,1):
    #             nn = NeuralNetwork(banknote_train, banknote_descriptor, layers=[4,N,N,1], function=functions, lr=1/(2**i), d_decay=1/(2**j), max_iters = 0, seed=2048)
    #             tr,te = test_model(nn, T, 1, banknote_train, banknote_test, y_train, y_test, error, title=f'N={N}, lr=2**-{i}, d=2**-{j}')
    #             results[(N,i,j)] = (tr[-1], te[-1])
    
    # print("Best found summary:")
    # best = {}
    # for N in NEURON_WIDTHS:
    #     val = 0
    #     for key in results:
    #         if key[0]==N:
    #             if not N in best or val > sum(results[key]):
    #                 best[N] = (key[1],key[2],*results[key])
    #                 val = sum(results[key])
    #     print("Best for N=",N,":",best[N])
       
    
    # Hard-coded result from tests
    best = {5: (2, 0, 0.0240825688073395, 0.040000000000000036),
      10: (3, 0, 0.020642201834862428, 0.040000000000000036),
      25: (3, 1, 0.0240825688073395, 0.03600000000000003),
      50: (2, 1, 0.019495412844036664, 0.03200000000000003),
      100: (2, 4, 0.01834862385321101, 0.03200000000000003)}
    
    # Now lets do more iterations
    T = 1000
    
    print("Testing on",T,"epochs")
    for N in NEURON_WIDTHS:
        i,j,_,_ = best[N]
        print(f'Training NN of width {N} with gamma0=2^-{i}, d=2^-{j}')
        nn = NeuralNetwork(banknote_train, banknote_descriptor, layers=[4,N,N,1], function=functions, lr=1/(2**i), d_decay=1/(2**j), max_iters = 0, seed=2048)
        tr,te = test_model(nn, T, 1, banknote_train, banknote_test, y_train, y_test, error, title=f'Final N={N}, lr=2**-{i}, d=2**-{j}',print_step=T//10)
        print(f'{N} & $2^{{-{i}}}$ & $2^{{-{j}}}$ & {tr[-1]:.4f} & {te[-1]:.4f} \\\\ ')
    
    print("Testing on",T,"epochs initializing w as 0")
    for N in NEURON_WIDTHS:
        i,j,_,_ = best[N]
        print(f'Training NN of width {N} with gamma0=2^-{i}, d=2^-{j}')
        nn = NeuralNetwork(banknote_train, banknote_descriptor, layers=[4,N,N,1], function=functions, lr=1/(2**i), d_decay=1/(2**j), max_iters = 0, seed=2048, weights='zero')
        tr,te = test_model(nn, T, 1, banknote_train, banknote_test, y_train, y_test, error, title=f'Final w=0, N={N}, lr=2**-{i}, d=2**-{j}',print_step=T//10)
        print(f'{N} & $2^{{-{i}}}$ & $2^{{-{j}}}$ & {tr[-1]:.4f} & {te[-1]:.4f} \\\\ ')
    