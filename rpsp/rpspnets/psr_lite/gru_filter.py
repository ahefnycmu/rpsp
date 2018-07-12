#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 21:02:29 2017

@author: ahefny
"""

from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T
       
from rpsp.rpspnets.psr_lite.rnn_filter import RNNFilter
from rpsp.rpspnets.psr_lite.utils.p3 import *

class GRUFilter(RNNFilter):
    #TODO: This is a hack
    def get_projs(self):
        return 0
    @property
    def _params_proj(self): return []
    @property
    def _opt_U(self): return 0.0
    @property
    def _opt_V(self): return 0.0
    
    def __init__(self, state_dim, hidden_out_dim, horizon, optimizer='sgd', optimizer_step=1.0,
                 optimizer_iterations=100, optimizer_min_step=1e-5, val_trajs=0, rng=None):                
        RNNFilter.__init__(self, state_dim, horizon, optimizer, optimizer_step,
                           optimizer_iterations, optimizer_min_step, val_trajs, rng=rng)
        
        self._use_hidden_output = (hidden_out_dim > 0)        
        self._hidden_out_dim = hidden_out_dim
                                                     
    def _init_params(self, traj_obs, traj_act):
        d_o = traj_obs[0].shape[1]
        d_a = traj_act[0].shape[1]
        d_h = self._state_dim
        d_g = self._hidden_out_dim if self._use_hidden_output else d_o
                
        s0 = self.rng.rand(d_h)*0.5-0.25
        self._t_state0 = theano.shared(name='state0',value=s0)
        
        r = -4*np.sqrt(6.0/(d_o+d_a+d_h))                        
        self._t_Winh_o = theano.shared(name='Winh_o',
                                      value=self.rng.rand(d_o,d_h)*(2*r)-r)
        self._t_Winh_a = theano.shared(name='Winh_a',
                                      value=self.rng.rand(d_a,d_h)*(2*r)-r)
        self._t_Winz_o = theano.shared(name='Winz_o',
                                      value=self.rng.rand(d_o,d_h)*(2*r)-r)
        self._t_Winz_a = theano.shared(name='Winz_a',
                                      value=self.rng.rand(d_a,d_h)*(2*r)-r)
        self._t_Winr_o = theano.shared(name='Winr_o',
                                      value=self.rng.rand(d_o,d_h)*(2*r)-r)
        self._t_Winr_a = theano.shared(name='Winr_a',
                                      value=self.rng.rand(d_a,d_h)*(2*r)-r)
        
        r = -4*np.sqrt(6.0/(d_h+d_h))                        
        self._t_Wh = theano.shared(name='Wh',
                                   value=np.eye(d_h) + self.rng.randn(d_h,d_h)*0.01)
        self._t_Wz = theano.shared(name='Wz',
                                   value=np.eye(d_h) + self.rng.randn(d_h,d_h)*0.01)
        self._t_Wr = theano.shared(name='Wr',
                                   value=np.eye(d_h) + self.rng.randn(d_h,d_h)*0.01)
        
        r = -4*np.sqrt(6.0/(d_h+d_g))
        self._t_W_hg = theano.shared(name='W_hg',
                                      value=self.rng.rand(d_h,d_g)*(2*r)-r)
        r = -4*np.sqrt(6.0/(d_a+d_g))
        self._t_W_ag = theano.shared(name='W_ag',
                                      value=self.rng.rand(d_a,d_g)*(2*r)-r)
        
        r = -4*np.sqrt(6.0/(d_h+d_o))
        self._t_W_out = theano.shared(name='W_out',
                                      value=self.rng.rand(d_g,d_o)*(2*r)-r)
        
        self._t_b_z = theano.shared(name='b_z', value=np.zeros(d_h))
        self._t_b_r = theano.shared(name='b_r', value=np.zeros(d_h))        
        self._t_b_h = theano.shared(name='b_h', value=np.zeros(d_h))
        self._t_b_g = theano.shared(name='b_g', value=np.zeros(d_g))
        self._t_b_out = theano.shared(name='b_out', value=np.zeros(d_o)) 
         
        self._params_state = [self._t_Winh_o,self._t_Winh_a,
                              self._t_Winz_o,self._t_Winz_a,
                              self._t_Winr_o,self._t_Winr_a,
                              self._t_Wh,self._t_Wz,self._t_Wr,
                              self._t_b_h,self._t_b_z,self._t_b_r]                             
        self._params_obs = [self._t_W_hg,self._t_W_ag,self._t_W_out,self._t_b_g,self._t_b_out]
        self._params_guide = self._params_obs
                             
                            
    def tf_update_state(self, tin_hm1, tin_o, tin_a):
        t_z = T.nnet.sigmoid(T.dot(tin_o,self._t_Winz_o) +
                             T.dot(tin_a,self._t_Winz_a) +
                             T.dot(tin_hm1,self._t_Wz) + self._t_b_z)
        
        t_r = T.nnet.sigmoid(T.dot(tin_o,self._t_Winr_o) +
                             T.dot(tin_a,self._t_Winr_a) +
                             T.dot(tin_hm1,self._t_Wr) + self._t_b_r)
        
        t_h = T.nnet.sigmoid(T.dot(tin_o,self._t_Winh_o) +
                             T.dot(tin_a,self._t_Winh_a) +
                             T.dot(tin_hm1*t_r,self._t_Wh) + self._t_b_h)
        
        tout_h = t_z*tin_hm1 + (1-t_z)*t_h
        
        return tout_h
            
    def tf_predict_obs(self, tin_h, tin_a):
        t_g = T.nnet.sigmoid(T.dot(tin_h, self._t_W_hg) + 
                          T.dot(tin_a, self._t_W_ag) + self._t_b_g)
        
        if self._use_hidden_output:
            tout_o = T.dot(t_g, self._t_W_out) + self._t_b_out
        else:
            tout_o = t_g
            
        return tout_o
    
