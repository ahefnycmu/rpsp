# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 16:02:23 2016

@author: ahefny
"""

import numpy as np
import numpy.linalg as npla

def ridge(X, Y, l2_lambda=0.0, importance_weights=None):
    Xw = X    
    if importance_weights is not None: Xw = X * np.sqrt(importance_weights.reshape((-1,1)))
            
    CC = np.dot(Xw.T,X)
    C = np.copy(CC)
    d = C.shape[0]
    C.ravel()[::d+1] += l2_lambda
    W = npla.solve(C, np.dot(Xw.T,Y))
    
    return W
    
    