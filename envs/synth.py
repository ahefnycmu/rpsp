#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:28:14 2017

@author: ahefny
"""

import numpy as np
from environments import Environment

'''
Simple synthetic environments for debugging
'''

class ReferenceValEnvironment(Environment):
    '''
    An environment where the agent is punished based on the magnitude of the action
    '''
    def __init__(self, rng):
        self._rng = rng

    def rng(self):
        return self._rng
        
    @property
    def dimensions(self):
        return (2,1)
            
    def render(self):
        pass
        
    def reset(self):
        return np.array([0.0, 0.0])
                
    def step(self, a):        
        o = self._rng.rand(2)
        r = -(a ** 2)
        return np.array(o), r, False
     
    def close(self):
        pass
    
class RepeatObsEnvironment(Environment):
    def rng():
        return 0.0

    @property
    def dimensions(self):
        return (1,1)
            
    def render(self):
        pass
        
    def reset(self):
        self._t = 0
        return self._f()

    def _f(self):
        t = self._t
        return np.cos(t/50.0)
        
    '''
    Given action a return a tuple:
    (observation, reward, is_episode_done)
    '''
    def step(self, a):
        self._t += 1
        o = self._f()
        r = -((o-a) ** 2)
        return o, r, False
     
    def close(self):
        pass
    


