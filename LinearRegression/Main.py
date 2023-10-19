# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

import numpy as np
import sys

from LMS import LMS
sys.path.append("../Preprocess")
from CSVLoader import CSVLoader
sys.path.append("../Postprocess")
from Metrics import *

try:
    import matplotlib.pyplot as plt
    plot = True
except ImportError:
    print("Missing matplotlib.pyplot")
    plot = False

concrete_train_path = 'concrete/train.csv'
concrete_test_path = 'concrete/test.csv'
concrete_descriptor = {
    'target': 'y',
    'columns': 'Cement,Slag,Fly ash,Water,SP,Coarse Aggr,Fine Aggr,y'.split(','),
    'numerical': 'Cement,Slag,Fly ash,Water,SP,Coarse Aggr,Fine Aggr,y'.split(',')
}

### MAIN
if __name__ == '__main__':
    concrete_train = CSVLoader(concrete_train_path, concrete_descriptor)
    concrete_test = CSVLoader(concrete_test_path, concrete_descriptor)
    
    # LMS Batch training
    print("LMS Batch training")
    lms = LMS(concrete_train, concrete_descriptor, 1/(2**7), threshold=1e-6, strategy='batch')
    if plot:
        plt.plot(lms.training_errors())
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.show()
    print("Learning rate:", lms.learning_rate())
    print("Weights:",lms.weights())
    
    y_train = get_y(concrete_train, concrete_descriptor)
    y_test = get_y(concrete_test, concrete_descriptor)
    
    pred_train = lms(concrete_train)
    pred_test = lms(concrete_test)
    
    print("Cost train:", square_error(y_train, pred_train))
    print("Cost test:", square_error(y_test, pred_test))
    
    print("\nLMS Stochastic training")
    lms2 = LMS(concrete_train, concrete_descriptor, 1/(2**9), threshold=1e-7, strategy='stochastic', seed=21)
    if plot:
        plt.plot(lms2.training_errors())
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.show()
    print("Learning rate:", lms2.learning_rate())
    print("Weights:",lms2.weights())
    
    pred_train = lms2(concrete_train)
    pred_test = lms2(concrete_test)
    
    print("Cost train:", square_error(y_train, pred_train))
    print("Cost test:", square_error(y_test, pred_test))
    
    
    print("\nAnalytical result:")
    print("Weights:", [np.array(lms.w)[i][0] for i in range(lms.w.shape[0]) ])
    