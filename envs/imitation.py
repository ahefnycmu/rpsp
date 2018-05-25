#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 00:42:02 2017

@author: ahefny
"""

from environments import RewardShaper
import numpy as np

class ImitiationEnvironment(RewardShaper):
    '''
    Wrapper for imitation learning using AggreVaTeD. The value function model
    is a function that accepts a state and time stamp and returns a value.
    '''
    def __init__(self, base_environment, value_function):        
        super(ImitiationEnvironment, self).__init__(base_environment)
        self._value_function = value_function
        
    def reset(self):        
        self._t = 0
        o = super(ImitiationEnvironment, self).reset()
        self._v_before = self._value_function(o, self._t)      
        return o
    
    def _reward_function(self, a, o_before, o_after, r):
        self._t += 1
        _v_after = self._value_function(o_after, self._t)
        #import pdb; pdb.set_trace()
        r = r + 0.99 * _v_after - self._v_before        
        #r = v_after
        self._v_before = _v_after
        return r

class Expert_Vf(object):
    def __init__(self, v_star_file_name, mean_std_file_name):
        from keras.models import load_model        
        self.net = load_model(v_star_file_name)
        npzfile = np.load(mean_std_file_name)
        self.mean = npzfile["arr_0"]
        self.std = npzfile["arr_1"]

    def __call__(self, raw_obs, time_step):
        obs = np.concatenate([raw_obs, np.array([time_step])],axis=0)
        norm_obs = (obs - self.mean)/self.std
        return self.net.predict(norm_obs.reshape(1,-1))[0][0]        
    
def load_value_function(hd5_file, npz_file):
    return Expert_Vf(hd5_file, npz_file) 
    

    
