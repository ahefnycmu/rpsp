# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 13:53:31 2016

@author: zmarinho
"""
import sys
import pdb
from IPython import embed
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cPickle as pickle
import os, time
import scipy.linalg
import psr_models.utils.linalg as lg
from distutils.dir_util import mkpath

DEBUG=False


class ridge_regression(object):
    def __init__(self, X,Y, reg=1e-6, add_const=False, imp_W=None):
        tic= time.time()
        n = X.shape[1]

        if add_const:
            Xf = np.concatenate( [np.ones((1,n),dtype=float), X] , axis=0)
            imp_W = np.concatenate( [np.ones((1,n),dtype=float), imp_W] , axis=0)
        else:
            Xf = X
        Xw = Xf
        if imp_W is not None:
            Xw = X*imp_W
        dx = Xf.shape[0]
        
        CovX = np.dot(Xw, Xf.T)
        CovXY = np.dot(Xw, Y.T)
        CovX[np.diag_indices(dx)] = CovX[np.diag_indices(dx)] + reg
        self.W = np.linalg.solve(CovX, CovXY).T
        if add_const:
            self.predict = lambda f: self.predict_const(self.W, f) 
        else:
            self.predict = lambda f: self.predict_simple(self.W, f)
        if DEBUG: print ('RR took: ', time.time()-tic)
        return 
    
    def predict_const(self, W, f):
        if f.ndim==1:
            f= f.reshape(-1,1)
        return np.dot(W, np.vstack([[1.0]*f.shape[1],f])).squeeze()
    
    def predict_simple(self, W, f):
        return np.dot(W, f)
        
class two_stage_regression(object):
    def __init__(self, X, Y, reg=1e-10, add_const=True):
        self.model_s1A = None
        self.model_s1B = None
        self.model_s2 = ridge_regression(X,Y,reg=reg,add_const=add_const)
        return
    
    def predict_ext(self, ext):
        return self.predict_simple(ext, self.W_s2ext) 
    
    def predict_oo(self, oo):
        return self.predict_simple(oo, self.W_s2oo) 
    
    
    
class extended_ridge_regression(ridge_regression):
    def __init__(self, X_o,X_a,Y, reg=1e-6,add_const=True):
        C_oa = lg.khatri_dot(X_a, X_o) 
        C_aa = lg.khatri_dot(X_a, X_a)
        dx = X_a.shape[0]
        super(extended_ridge_regression, self).__init__(C_aa,C_oa,reg=reg,add_const=add_const)
        return
        
class joint_ridge_regression(ridge_regression):
    def __init__(self, Xs, instrument, func=None, add_const=False, reg=1e-10, imp_W=None):
        self.split_loc = [0]
        X = self.join_output(Xs, func=func)
        super(joint_ridge_regression, self).__init__(instrument, X, reg=reg, add_const=add_const, imp_W=imp_W)
        return
    
    def split_output(self, X ):
        Xs =[]
        locs = np.cumsum(self.split_loc)
        for i in xrange(len(self.split_loc)-1):
            start = locs[i]
            end = locs[i+1]
            Xs.append( X[start:end,:] )
        return Xs
    
    def join_output(self, Xs, func=None):
        X = []
        for i in xrange(0,len(Xs),2):
            Coa, Caa = func(Xs[i], Xs[i+1])
            X.extend([Coa, Caa])
            self.split_loc.extend( [Coa.shape[0], Caa.shape[0]] )
        return np.concatenate(X, axis=0)
    
    
def khatri_pair(X_o,X_a):
    C_oa = lg.khatri_dot(X_a, X_o)
    C_aa = lg.khatri_dot(X_a, X_a)
    return C_oa,C_aa

def denoise_data(X, instrument, trainer, func=None):
    N = instrument.shape[1]
    model = trainer(X, instrument, func=func)
    ef = model.predict(instrument[:,0])
    d = ef.shape[0]
    Xbar = np.zeros((d,N), dtype=float)
    
    for i in xrange(N):
        Xbar[:,i] = model.predict( instrument[:,i])        
    return Xbar, model

        
def denoise_cov(X, instrument, reg=1e-6, trainer=joint_ridge_regression, func=khatri_pair, const=False, imp_W=None):
    N = instrument.shape[1]
    model = trainer(X, instrument, func=func, add_const=const, reg=reg, imp_W=imp_W)
    d = model.predict(instrument[:,0]).shape[0]
    denoised_op = np.zeros((d,N), dtype=float)
    
    for i in xrange(N):
        denoised_op[:,i] = model.predict( instrument[:,i])     
    Coa, Caa = model.split_output(denoised_op)
    return Coa, Caa
        
def  train_two_stage_model( input, output, instrument, s1a_trainer, s1b_trainer, s2_trainer):
    ''' TRAIN_TWO_STAGE_MODEL    Fits data using two-stage "instrument" regression.
    % The model predicts "output" from "input" using "ins" to cancel correlated
    % noise. In time series context, "input" == future (i.e. belief state),
    % "output" == extended future (i.e. belief state + observation) and
    % "instrument" == past.
    %   Paramaters:
    %       input, output, instrument - Training set in column order format
    %       s1a_trainer - Regression model for S1A.
    %       s1b_trainer - Regression model for S1B.
    %       s2_trainer - Regression model for S2.
    '''
    print('Stage 1 Regression')
    input_denoise, model_s1a = denoise_data(input, instrument, s1a_trainer) # [1; input]
    output_denoise, model_s1b = denoise_data(output, instrument, s1b_trainer) #[1; output]
    
    print('Stage 2 Regression')
    model = s2_trainer(input_denoise, output_denoise)
    model.model_s1A = model_s1a
    model.model_s1B = model_s1b

    return model, input_denoise, output_denoise  




  
def test_ridge_regression(args):
    np.random.seed(100)
    
    ''' Test for ridge regression functions:
       train_ridge_regression
       train_ridge_regression_lowmem
       train_ridge_regression_highdim'''
    
    W = np.array([[1, 0.5,0,0],[ 0.1, 2,0,0],[0,0,1, 0.5],[ 0,0,0.1, 2]],dtype=float)
    N = 1000
    d=4
    X = np.random.randn(d, N)
    Y = np.dot(W,X) + np.random.randn(d, N) * 1e-3
    n1=2
    n2=2
    
    # Train regression models
    model = ridge_regression(X, Y, 1e-10, False);

    embed()
    X1, X2 = denoise_cov([X[:2,:], X[2:,:]], Y)
    Yp2 = np.zeros((d, N),dtype=float)
    Yp3 = np.zeros((d, N),dtype=float)
    for t in xrange(N):
        C_bb = X2[:,t].reshape((n2,n2),order='F')
        C_ab = X1[:,t].reshape((n1,n2), order='F')
        Yp2[:,t] = ridge_regression( C_bb, C_ab, 1e-10,False).W.reshape(-1,order='F')
        Yp3[:,t] = lg.reg_rdivide(np.dot(C_ab,C_bb),np.dot(C_bb,C_bb),1e-10).reshape(-1,order='F')
    
    print np.linalg.norm(Yp2-Yp3)
    
    # Test regression models and report parameter error and training MSE
    print('Parameter difference:')
    print(model.W - W)

    Yp = np.zeros((d, N),dtype=float)
    
    for t in xrange(N):
        Yp[:,t] = model.predict( X[:2,t])
    

    E = Yp - Y
    E = np.sum(E * E, 0)
    nrm = np.sum(Y * Y, 0)

    mse = sum(E) / N
    print('MSE: %f\n', mse);            

    
    assert(np.max(mse) < 1e-5),'high error'
    embed()
    return




if __name__=='__main__':
    test_ridge_regression(sys.argv[1:])