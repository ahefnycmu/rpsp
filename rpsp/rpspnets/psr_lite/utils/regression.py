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


class RidgeRegression(object):

    def __init__(self, Ridge, x_dim=None, y_dim=None):
        self.ridge = Ridge;
        self.A = None;

        if x_dim is not None and y_dim is not None:
            self.initialize(x_dim, y_dim);

    def initialized(self):
        if self.A is None:
            return False;
        else:
            return True;

    def get_params(self):
        return self.A;

    def set_params(self, A):
        self.A = A;

    def initialize(self, x_dim, y_dim):
        # self.A = np.random.randn(x_dim, y_dim) * 2.0/np.sqrt(x_dim + y_dim);
        self.A = np.zeros((x_dim + 1, y_dim));  # x plus interception.

    def fit(self, input_X, Y):
        X = np.hstack((input_X, np.ones((input_X.shape[0], 1))));
        self.dy = Y.shape[1];
        N = X.shape[0];
        self.A = np.dot(np.dot(Y.T, X), np.linalg.inv(np.dot(X.T, X) + N * self.ridge * np.identity(X.shape[1]))).T;

    def predict(self, input_X):
        X = np.hstack((input_X, np.ones((input_X.shape[0], 1))));
        # if self.A.shape[0] == 0:
        #	self.A = np.zeros(X.shape[1], self.dy);
        Yhat = np.dot(X, self.A);
        return Yhat;