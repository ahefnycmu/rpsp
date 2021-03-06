# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 20:24:37 2016

@author: ahefny, zmarinho
"""

from p3 import *

import numpy as np
from IPython import embed

def onehot2ind(ind_v):
    a = np.argmax(ind_v,axis=0)
    return a

def ind2onehot(ind_v, dim):
    ''' convert to one hot representation if dim<>None otherwise return original vector'''
    if dim is None:
        return ind_v
    if np.isscalar(ind_v):
        ind_v = np.asarray([[int(ind_v)]])
    elif type(ind_v) == list:
        ind_v = np.asarray(ind_v).reshape(1,-1)
    elif ind_v.ndim==1:
        ind_v = np.asarray([ind_v])
    n = ind_v.shape[1]
    x = np.zeros((dim, n),dtype=float)
    x[np.array(ind_v, dtype=int), np.arange(n)] = 1.0
    return x.squeeze()


def timewin_features(X, t, win_length, delta):
    T,d = X.shape
    t += delta
        
    assert t >= 0 and t+win_length <= T ,'bad lag in timewin features'
    return X[t:t+win_length,:].reshape(d*win_length)

def finite_past_feat_extractor(past_length, lag=1):
    return lambda X,t: timewin_features(X, t, past_length, -past_length-lag+1)
    
def finite_future_feat_extractor(future_length, lag=0):
    return lambda X,t: timewin_features(X, t, future_length, lag)    
    
def validate_seqs(X, bounds,Lmin=2):
    newX = [X[i] for i in xrange(len(X)) if X[i].shape[0] > (bounds[0]+bounds[1]+Lmin) ] #need only 2
    return newX
    
def flatten_features(X, feat_extractor, bounds=(0,0)):
    newX = validate_seqs(X, bounds)
    assert len(newX)>0, 'too small trajectory in flatten features!'
    X = newX
    
    N = sum(X[i].shape[0] for i in xrange(len(X))) - (bounds[0]+bounds[1]) * len(X)    
    d = feat_extractor(X[0],bounds[0]).size
    Y = np.zeros((N,d))
    
    series_index = np.zeros(N)
    time_index = np.zeros(N)

    n = 0    
    for i in xrange(len(X)):
        start = bounds[0]
        end = X[i].shape[0]-bounds[1]
        
        for j in xrange(start,end):
            series_index[n] = i
            time_index[n] = j
            Y[n,:] = feat_extractor(X[i], j)
            n += 1
    
    #TODO: fix series index to reflect remove time series.
    #return (Y, series_index, time_index)
    return (Y, None, time_index)
    
def normalize(X, std=1):            
    mu = np.mean(X, axis=0)
    X -= mu
    
    if std == 1:
        s = np.std(X, axis=0)
        X /= s                
        X[:,np.where(s == 0)] = 0.0    
    
if __name__ == '__main__':
    def test_timewin_features():
        a = np.array(xrange(10))    
        b = np.array(xrange(9,-1,-1))
        X = np.vstack((a,b)).T
        
        p = 2
        f = 2
        
        p_feat = finite_past_feat_extractor(p)
        f_feat = finite_future_feat_extractor(f)
        
        assert np.all(p_feat(X,2) == np.array([0,9,1,8]))
        assert np.all(f_feat(X,1) == np.array([1,8,2,7]))
        
        XX = [X, X[:5,:]]                
        Y,si,ti = flatten_features(XX, p_feat, bounds=(p,0))
                        
        assert np.all(Y[0,:] == Y[8,:])
        assert np.all(si == np.hstack((np.ones(8) * 0, np.ones(3) * 1)))
        assert np.all(ti == np.hstack((np.array(xrange(2,10)), np.array(xrange(2,5)))))
        
    
    test_timewin_features()
    
    
    
    
