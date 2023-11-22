# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

import numpy as np
import sys

from SVMSGD import SVMSGD,decay
from SVM import SVM

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

def test_model(model, T, step, x_train, x_test, y_train, y_test, error_function, do_plot=True, title=''):
    train_errors = []
    test_errors = []
    for i in range(1,T+1,step):
        model.modify_epochs(i)
        train_pred = model(x_train)
        test_pred = model(x_test)
        train_errors.append(error_function(y_train,train_pred))
        test_errors.append(error_function(y_test,test_pred))
        print("T=",i,train_errors[-1],test_errors[-1])
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

def beauty_weight(w, separator=', '):
    return separator.join(f'{i:.4f}' for i in w)

### MAIN
if __name__ == '__main__':
    x = np.array([[0.5, -1, 0.3, 1], [-1,-2,-2, 1], [1.5,0.2,-2.5, 1]])
    y = np.array([1,-1,1])
    w = np.array([0,0,0,0])
    lr = [0.01, 0.005, 0.0025]
    C = 1
    N = len(x)

    res = ''

    for i in range(3):
        r = y[i] * np.dot(x[i], w)
        w0 = w.copy()
        w0[-1] = 0
        gamma = lr[i]
        res += f'{i+1} & {gamma} & '
        if r <= 1:
            d = w0 - C*N*y[i]*x[i]
            # res += f'({d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f}, {d[3]:.3f}) & '
            d = gamma*w0 - gamma*C*N*y[i]*x[i]
            res += f'({d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f}, {d[3]:.3f})'
            w = w - gamma*w0 + gamma*C*N*y[i]*x[i]
        else:
            d = - gamma*w0
            res += f'({d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f}, 0)'
            w[:3] = (1 - gamma)*w0

        res += f' & ({w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}, {w[3]:.3f}) \\\\\n'
    # print(res)
    
    banknote_train = CSVLoader(banknote_train_path, banknote_descriptor)
    banknote_test = CSVLoader(banknote_test_path, banknote_descriptor)
    
    banknote_train = replace_value(banknote_train, banknote_descriptor, 'genuine', 0, -1.0)
    banknote_test = replace_value(banknote_test, banknote_descriptor, 'genuine', 0, -1.0)
    
    y_train = get_y(banknote_train, banknote_descriptor)
    y_test = get_y(banknote_test, banknote_descriptor)
    
    T = 100
    
    print("Evaluating SVM SGD with a in decay")
    # print("Tuning for C=100/873")
    # for i in range(1,15):
    #     for j in range(1,i+6):
    #         svm = SVMSGD(banknote_train, banknote_descriptor, C=100/873, lr=1/(2**i), lr_function=decay, a=1/(2**j), max_iters = 0, seed=2048)
    #         test_model(svm, T, 1, banknote_train, banknote_test, y_train, y_test, error, title=f'lr=2**-{i}, a=2**-{j}')
    
    # print("Tuning for C=500/873")
    # for i in range(1,15):
    #     for j in range(1,i+6):
    #         svm = SVMSGD(banknote_train, banknote_descriptor, C=500/873, lr=1/(2**i), lr_function=decay, a=1/(2**j), max_iters = 0, seed=2048)
    #         test_model(svm, T, 1, banknote_train, banknote_test, y_train, y_test, error, title=f'lr=2**-{i}, a=2**-{j}')
    
    # print("Tuning for C=700/873")
    # for i in range(1,15):
    #     for j in range(1,i+6):
    #         svm = SVMSGD(banknote_train, banknote_descriptor, C=700/873, lr=1/(2**i), lr_function=decay, a=1/(2**j), max_iters = 0, seed=2048)
    #         test_model(svm, T, 1, banknote_train, banknote_test, y_train, y_test, error, title=f'lr=2**-{i}, a=2**-{j}')
    
    lr = 1/(2**8)
    a = 1/(2**12)
    
    w_sgda = []
    print("\nFinal model for SVM SGD with a in decay")
    for C in [100, 500, 700]:
        svm = SVMSGD(banknote_train, banknote_descriptor, C=C/873, lr=lr, lr_function=decay, a=a, max_iters = T, seed=2048)
        train_pred = svm(banknote_train)
        test_pred = svm(banknote_test)
        print(f"C={C}/873",error(y_train,train_pred),error(y_test,test_pred))
        print("w+b", beauty_weight(svm.get_weights(), ', '))
        w_sgda.append(svm.get_weights())
    
        
    
    print("\nEvaluating SVM SGD without a in decay")
    # print("Tuning for C=100/873")
    # for i in range(1,16):
    #     svm = SVMSGD(banknote_train, banknote_descriptor, C=100/873, lr=1/(2**i), lr_function=decay, a=None, max_iters = 0, seed=2048)
    #     test_model(svm, T, 1, banknote_train, banknote_test, y_train, y_test, error, title=f'lr=2**-{i}')
    
    # print("Tuning for C=500/873")
    # for i in range(1,16):
    #     svm = SVMSGD(banknote_train, banknote_descriptor, C=500/873, lr=1/(2**i), lr_function=decay, a=None, max_iters = 0, seed=2048)
    #     test_model(svm, T, 1, banknote_train, banknote_test, y_train, y_test, error, title=f'lr=2**-{i}')
    
    # print("Tuning for C=700/873")
    # for i in range(1,16):
    #     svm = SVMSGD(banknote_train, banknote_descriptor, C=700/873, lr=1/(2**i), lr_function=decay, a=None, max_iters = 0, seed=2048)
    #     test_model(svm, T, 1, banknote_train, banknote_test, y_train, y_test, error, title=f'lr=2**-{i}')
    
    lr = 1/(2**15)
    
    w_sgd = []
    print("\nFinal model for SVM SGD without a in decay")
    for C in [100, 500, 700]:
        svm = SVMSGD(banknote_train, banknote_descriptor, C=C/873, lr=lr, lr_function=decay, a=None, max_iters = T, seed=2048)
        train_pred = svm(banknote_train)
        test_pred = svm(banknote_test)
        print(f"C={C}/873",error(y_train,train_pred),error(y_test,test_pred))
        print("w+b", beauty_weight(svm.get_weights(), ', '))
        w_sgd.append(svm.get_weights())
    
    ws = []
    print("\nSVM dual form")
    for C in [100, 500, 700]:
        svm = SVM(banknote_train, banknote_descriptor, C=C/873)
        train_pred = svm(banknote_train)
        test_pred = svm(banknote_test)
        print(f"C={C}/873",error(y_train,train_pred),error(y_test,test_pred))
        print("w+b", beauty_weight(svm.get_weights(), ', '))
        ws.append(svm.get_weights())
    
    print("\nRelationship between w with obtained w from SVM-dual form")
    for C in [100, 500, 700]:
        sgda = np.array(w_sgda[i])
        sgd = np.array(w_sgd[i])
        w = np.array(ws[i])
        print(f"C={C}, SGDA:", sgda/w)
        print(f"C={C}, SGD:", sgd/w)
        
    print("\nSVM dual form with gaussian kernel")
    no_support_vecs = []
    for C in [100, 500, 700]:
        for gamma in [0.1, 0.5, 1,  5, 100]:
            svm = SVM(banknote_train, banknote_descriptor, C=C/873, kernel='gaussian', gamma=gamma)
            train_pred = svm(banknote_train)
            test_pred = svm(banknote_test)
            print(f"C={C}/873, gamma={gamma}",error(y_train,train_pred),error(y_test,test_pred))
            print("w+b", beauty_weight(svm.get_weights(), ', '))
            # print(f"{C}/873 & {gamma} & {error(y_train,train_pred):.4f} & {error(y_test,test_pred):.4f} & ({beauty_weight(svm.get_weights()[:-1], ', ')}) & {beauty_weight(svm.get_weights()[-1:], ', ')} \\\\")
            no_support_vecs.append((C,gamma,len(svm.get_support_vecs())))
    
    print(no_support_vecs)
    
    repeated_support_vecs = []
    C = 500
    for gamma in [0.01, 0.1, 0.5]:
        svm = SVM(banknote_train, banknote_descriptor, C=C/873, kernel='gaussian', gamma=gamma)
        repeated_support_vecs.append(svm.get_support_vecs())
    
    gamma01 = set(tuple(i) for i in repeated_support_vecs[0])
    gamma1 = set(tuple(i) for i in repeated_support_vecs[1])
    gamma5 = set(tuple(i) for i in repeated_support_vecs[2])
    
    intersect011 = gamma01 & gamma1
    intersect15 = gamma5 & gamma1
    
    print("Support vectors:", len(gamma01), len(gamma1), len(gamma5))
    print("Intersection/repeated vectors:", len(intersect011), len(intersect15))
