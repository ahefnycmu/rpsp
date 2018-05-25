#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 18:44:19 2016

@author: ahefny
"""
from utils.p3 import *

import numpy as np
from rffpsr import ControlledModel, UncontrolledModel

def square_error(x,y):
    diff = x - y
    return np.sum(diff * diff)
    
def l1_error(x,y):
    diff = x - y    
    return np.sum(np.abs(diff))    
    
def clamped_square_error(x,y,min_x,max_x):
    z = np.maximum(np.minimum(x,max_x), min_x)
    return square_error(z,y)

def clamped_l1_error(x,y,min_x,max_x):
    z = np.maximum(np.minimum(x,max_x), min_x)
    return l1_error(z,y)
    
def run_psr_predict_horizon(psr, obs, act, initial_state=None, burn_in=0, err_func=square_error):    
    if initial_state is None: initial_state = psr.initial_state
    
    N,d = obs.shape
    k = psr.horizon_length
    M = N-k
    
    s = initial_state
    states = np.empty((M,s.size))
    est_obs = np.empty((k,M,d))    
    errors = np.empty((M-burn_in,k))
                
    for i in xrange(M):
        if isinstance(psr, ControlledModel):
            oh = psr.predict_horizon(s, act[i:i+k,:])
            s = psr.update_state(s, obs[i,:], act[i,:])
        else:
            assert isinstance(psr, UncontrolledModel)            
            oh = psr.predict_horizon(s)
            s = psr.update_state(s, obs[i,:])
            
        states[i,:] = s.ravel()
        est_obs[:,i,:] = oh

        if i >= burn_in:
            for j in xrange(k):                   
                errors[i-burn_in,j] = err_func(oh[j,:], obs[i+j,:])
                                
    return np.mean(errors,0), est_obs, states

def eval_psr(psr, X_tst_obs, X_tst_act, err_func=square_error, burn_in=10):
    mse = np.zeros(psr.horizon_length)
    N_tst = len(X_tst_obs)
    
    for i in xrange(N_tst):    
        mse_i,xh,s = run_psr_predict_horizon(
             psr, X_tst_obs[i], X_tst_act[i], err_func=err_func, burn_in=burn_in)
        mse += mse_i 
        
    mse /= N_tst
    return mse    
    