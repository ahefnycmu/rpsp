#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:59:34 2017

@author: ahefny
"""
from utils.p3 import *

import numpy as np
import utils.feats
from utils.regression import ridge
from feat_extractor import FeatureExtractor

class structtype():
    pass

class UncontrolledModel(object):
    def train(self, traj_obs):
        raise NotImplementedError
            
    def update_state(self, state, obs):
        raise NotImplementedError
        
    def predict_shifted_state(self, state):
        raise NotImplementedError
        
    def predict_obs(self, state):
        raise NotImplementedError
        
    def predict_horizon(self, state):
        raise NotImplementedError
                        
    @property
    def state_dimension(self):
        raise NotImplementedError        
        
    @property
    def horizon_length(self):
        raise NotImplementedError

    @property
    def initial_state(self):
        raise NotImplementedError

class ControlledModel(object):
    def train(self, traj_obs, traj_act, traj_act_probs=None):
        raise NotImplementedError
            
    '''
    Update the state based on doing act and then observing obs
    '''
    def update_state(self, state, obs, act):
        raise NotImplementedError
        
    def predict_shifted_state(self, state, act):
        raise NotImplementedError
        
    def predict_obs(self, state, act):
        raise NotImplementedError
        
    def predict_horizon(self, state, fut_act):
        # Default implementation: recusrive prediction
        h, d_a = fut_act.shape
        a = fut_act[0,:]
        o = self.predict_obs(state, a)

        output = np.empty((h, o.size))
        output[0,:] = o

        for i in xrange(1,h):
            state = self.update_state(state, o, fut_act[i-1,:])            
            o = self.predict_obs(state, fut_act[i,:])
            output[i,:] = o   

        return output                     
                        
    @property
    def compute_pre_states(self):
        raise NotImplementedError                      
    @property
    def state_dimension(self):
        raise NotImplementedError        
        
    @property
    def horizon_length(self):
        raise NotImplementedError

    @property
    def initial_state(self):
        raise NotImplementedError                
        
def extract_timewins(traj_obs, traj_act, fut, past):       
    data = structtype()
    d = structtype()

    bounds = (past, fut)                
    past_extractor = utils.feats.finite_past_feat_extractor(past)
    shifted_past_extractor = utils.feats.finite_past_feat_extractor(past,0)
    fut_extractor = utils.feats.finite_future_feat_extractor(fut)
    shifted_fut_extractor = utils.feats.finite_future_feat_extractor(fut, 1)
    extended_fut_extractor = utils.feats.finite_future_feat_extractor(fut+1)
    immediate_extractor = lambda X,t: X[t,:]
                    
    data.past_obs, data.series_index, data.time_index = \
        utils.feats.flatten_features(traj_obs, past_extractor, bounds)
    data.past_act,_,_ = utils.feats.flatten_features(traj_act, past_extractor, bounds)
    data.past = np.hstack((data.past_obs, data.past_act))
    data.shpast_obs,_,_ = utils.feats.flatten_features(traj_obs, shifted_past_extractor, bounds)
    data.shpast_act,_,_ = utils.feats.flatten_features(traj_act, shifted_past_extractor, bounds)
    data.shpast = np.hstack((data.shpast_obs, data.shpast_act))
    data.fut_obs,_,_ = utils.feats.flatten_features(traj_obs, fut_extractor, bounds)
    data.fut_act,_,_ = utils.feats.flatten_features(traj_act, fut_extractor, bounds)
    data.shfut_obs,_,_ = utils.feats.flatten_features(traj_obs, shifted_fut_extractor, bounds)
    data.shfut_act,_,_ = utils.feats.flatten_features(traj_act, shifted_fut_extractor, bounds)
    data.exfut_act,_,_ = utils.feats.flatten_features(traj_act, extended_fut_extractor, bounds)
    data.obs,_,_ = utils.feats.flatten_features(traj_obs, immediate_extractor, bounds)
    data.act,_,_ = utils.feats.flatten_features(traj_act, immediate_extractor, bounds)

    d.h = data.past.shape[1]
    d.o = data.obs.shape[1]
    d.a = data.act.shape[1]
    d.fo = data.fut_obs.shape[1]
    d.fa = data.fut_act.shape[1]
                       
    return data, d
        
class AutoRegressiveControlledModel(ControlledModel):
    def __init__(self, fut, past, past_feats=None, l2_lambda=1e-3):
        if past_feats is None:
            past_feats = FeatureExtractor()

        self._lambda = l2_lambda
        self._fut = fut
        self._past = past                
        self._feats = past_feats
    
    def train(self, traj_obs, traj_act, traj_act_probs=None):
        if traj_act_probs is not None:
            raise 'Non-blind policies not supported'
            
        data, self._d = extract_timewins(traj_obs, traj_act, self._fut, self._past)
        input = np.hstack([data.past, data.fut_act])
        self._feats.build(input)                
        self._W = ridge(self._feats.process(input), data.fut_obs, l2_lambda=self._lambda)            
        self._f0 = np.mean(data.past, axis=0)        
            
    def update_state(self, state, obs, act):
        do = self._d.o
        da = self._d.a
        p = self._past
        
        state[:do*(p-1)] = state[do:do*p]
        state[do*(p-1):do*p] = obs
        state[do*p:do*p+da*(p-1)] = state[do*p+da:do*p+da*p]
        state[do*p+da*(p-1):do*p+da*p] = act
        return state                              
        
    def predict_shifted_state(self, state, act):
        raise NotImplementedError
        
    def predict_obs(self, state, act):
        raise NotImplementedError
        
    def predict_horizon(self, state, fut_act):
        input = np.hstack([state, fut_act.ravel()])
        return np.dot(self._feats.process(input), self._W).reshape((fut_act.shape[0],-1))
                        
    @property
    def state_dimension(self):
        return self._d.past        
        
    @property
    def horizon_length(self):
        return self._fut

    @property
    def initial_state(self):
        return self._f0
    
    '''
    Compute all states s.t. state[t] is after doing action[t]
    and then observing obs[t]
    '''
    def compute_pre_states(self, traj):
        # Use scan function
        state = self._f0
        states = []
        for i in xrange(len(traj.obs)):
            state = self.update_state(state, traj.obs[i], traj.act[i])
            states.append(state)
        return np.vstack(states)

class LastObsModel(ControlledModel):
    def __init__(self, obs_dim, horizon):
        self._q0 = np.zeros(obs_dim)
        self._fut = horizon
                
    def train(self, traj_obs, traj_act, traj_act_probs=None):
        pass
        
    def update_state(self, state, obs, act):        
        return obs                              
        
    def predict_shifted_state(self, state, act):
        return state
        
    def predict_obs(self, state, act):
        return state
        
    def predict_horizon(self, state, fut_act):
        return np.tile(state, (self._fut,1))        
                        
    @property
    def state_dimension(self):
        return self._d.past        
        
    @property
    def horizon_length(self):
        return self._fut

    @property
    def initial_state(self):
        return self._q0
    
    '''
    Compute all states s.t. state[t] is after doing action[t]
    and then observing obs[t]
    '''
    def compute_pre_states(self, traj):
        # Use scan function
        state = self._f0
        states = []
        for i in xrange(len(traj.obs)):
            state = self.update_state(state, traj.obs[i], traj.act[i])
            states.append(state)
        return np.vstack(states)     
