#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 23:00:03 2017

@author: ahefny
"""

from rpsp.rpspnets.psr_lite.psr_base import ControlledModel, extract_timewins
from rpsp.rpspnets.psr_lite.utils.kernel import median_bandwidth, gram_matrix_rbf
import numpy as np
import numpy.linalg as npla

class HSEPSR(ControlledModel):
    def __init__(self, fut, past, s=None, l2_lambda=1e-3, rng=None):
        if rng is None:
            from utils.misc import get_default_rand 
            rng = get_default_rand()
        
        self._fut = fut
        self._past = past
        self._rng = rng
        self._s = s
        self._lambda = l2_lambda
    
    def train(self, traj_obs, traj_act, traj_act_probs=None):
        data, d = extract_timewins(traj_obs, traj_act, self._fut, self._past)
        N = len(data.obs)
        self._N = N

        if self._s is None:        
            # Compute kernel bandwidths using merdian trick        
            s_past = median_bandwidth(data.past, 5000, self._rng)
            s_fut_o = median_bandwidth(data.fut_obs, 5000, self._rng)
            s_fut_a = median_bandwidth(data.fut_act, 5000, self._rng)
            s_o = median_bandwidth(data.obs, 5000, self._rng)
            s_a = median_bandwidth(data.act, 5000, self._rng)
        else:
            s_past = self._s[0]
            s_fut_o = self._s[1]
            s_fut_a = self._s[2]
            s_o = self._s[3]
            s_a = self._s[4]
        
        #test_gram = False
        #if test_gram:
        #    gram_matrix_rbf = lambda X,Y,*args: X.dot(Y.T)
        
        # Compute Gram Matrices
        G_h = gram_matrix_rbf(data.past, data.past, s_past)  
        G_hsh = gram_matrix_rbf(data.past, data.shpast, s_past)  
        G_o = gram_matrix_rbf(data.obs, data.obs, s_o)           
        G_a = gram_matrix_rbf(data.act, data.act, s_a)
        
        G_fo = gram_matrix_rbf(data.fut_obs, data.fut_obs, s_fut_o)        
        G_fa = gram_matrix_rbf(data.fut_act, data.fut_act, s_fut_a)
        
        # S1 Regression
        R = self._lambda*N*np.eye(N)
        Ah = npla.solve(G_h + R, G_h)
        Ash = npla.solve(G_h + R, G_hsh)
        
        # Compaute State gram matrices
        # This is an N^3 storage, N^4 runtime method    
        GGo = np.zeros((N*N,N))
        GGa = np.zeros((N*N,N))
        Gah = np.zeros((N,N,N))
                      
        for i in xrange(N):
            print(i)
            ai = Ah[:,i].reshape((-1,1))            
            Gai = npla.solve((ai * G_fa) + R, np.diag(ai.ravel()))
            GGa[:,i] = Gai.dot(G_fa).ravel()
            GGo[:,i] = G_fo.dot(Gai).ravel()
            Gah[:,:,i] = Gai
            
        G_s = GGa.T.dot(GGo) # state * state  
        
        # Shifted State
        for i in xrange(N):
            print('%d - shifted' % i)                
            ai = Ash[:,i].reshape((-1,1))
            Gai = npla.solve((ai * G_fa) + R, np.diag(ai.ravel()))           
            GGo[:,i] = G_fo.dot(Gai).ravel()
        
        G_ss = GGa.T.dot(GGo) # state * shifted state
        
        # S2 Regression
        '''
        This is a matrix that accepts a vector of weights of shifted futures
        (i.e. the outcome of the previous filtering step)
        and produces a vector of weights of extended futures.
        '''
        W2 = npla.solve(G_s + self._lambda * N * np.eye(N), G_ss)
            
        self._W2 = W2
        self._G_a = G_a
        self._G_o = G_o
        self._G_fa = G_fa
        self._G_fo = G_fo
        self._Gah = Gah
        self._Ah = Ah
        
        self._data = data
        self._s_a = s_a
        self._s_fut_a = s_fut_a
        self._s_o = s_o
        
        self._f0 = np.ones(N)/N 
            
    '''
    Update the state based on doing act and then observing obs
    '''
    def update_state(self, state, obs, act):        
        N = len(self._data.obs)
        obs = obs.reshape((-1,1))
        act = act.reshape((-1,1))
        
        # Apply S2 Regression
        q = self._W2.dot(state)
        
        # Filter Action        
        G = np.reshape(self._Gah,(N*N,N)).dot(q).reshape((N,N))        
        
        k_a = gram_matrix_rbf(self._data.act, act, self._s_a)
        q_a = G.dot(k_a)
        
        # Filter Observations
        G = q_a.reshape((-1,1)) * self._G_o
        k_o = gram_matrix_rbf(self._data.obs, obs, self._s_o)
        qp = npla.solve(G + self._lambda * N * np.eye(N), (q_a * k_o))
        return qp        
        
                    
    def predict_shifted_state(self, state, act):
        raise NotImplementedError
        
    def predict_obs(self, state, act):
        raise NotImplementedError
        
    def predict_horizon(self, state, fut_act):
        fa = fut_act.reshape((1,-1))
        all_future_act = self._data.fut_act
        all_fut_obs = self._data.fut_obs    
        s_fa = self._s_fut_a;                
        N = len(all_future_act)

        # Apply S2 regression (to express state as a weight of current -instead of shifted- futures)
        q = self._W2.dot(state)

        # Condition on actions
        G = self._Gah.reshape((N*N,N)).dot(q).reshape((N,N))        
        k_a = gram_matrix_rbf(all_future_act, fa, s_fa)
        q_fa = G.dot(k_a)
                       
        o = q_fa.ravel().dot(all_fut_obs)
        o = o.reshape((len(fut_act),-1))
        return o
                        
    @property
    def compute_pre_states(self):
        raise NotImplementedError                      
    @property
    def state_dimension(self):
        return self._N
        
    @property
    def horizon_length(self):
        return self._fut

    @property
    def initial_state(self):
        return self._f0
