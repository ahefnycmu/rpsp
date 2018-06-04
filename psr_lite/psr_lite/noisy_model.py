#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 12:38:44 2017

@author: ahefny, zmarinho
"""

from theano.tensor.shared_randomstreams import RandomStreams

'''
decorator of noisy_model.
'''   
class NoisyModel(object):        
    def __init__(self, obs_noise=0.0, obs_loc=0.0, state_noise=0.0, state_loc=0.0,
                  state_dim=0, rng=None):
        self._srng = RandomStreams(seed=rng.seed())
        self.rng = rng
        self._obs_loc = obs_loc
        self._state_loc = state_loc
        self._obs_std = obs_noise
        self._state_std = state_noise
        self._state_dim = state_dim
        self._state_noise =  self._srng.normal(size=[self._state_dim], std=obs_noise, avg=state_loc)
       
    def _noisy_state(self, state):
        if self._state_std>0:
            state = state + self._state_noise 
        return state

    def _noisy_obs(self, obs):
        noise = 0.0 
        if self._obs_std>0:
            noise = self.rng.normal(loc=self._obs_loc, scale=self._obs_std, size=obs.shape) 
        o = obs + noise
        return o 
