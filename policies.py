# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 10:47:38 2016

@author: ahefny

Policies are BLIND to the representation of states, which could be (1) observation, 
(2) original latent state or (3) predictive state. 

Policies takes the "state" dimension x_dim, the number of actions/dim of action as input.

"""

import numpy as np
import scipy.stats

class BasePolicy(object):
    def reset(self):
        pass
        
    def sample_action(self, state):
        '''
        Samples an action and returns a tuple consisting of:
            chosen action, action probability,
            dictionary of diagnostic information (values must be numbers or vectors)            
        '''
        raise NotImplementedError
    
    def _load(self, params):
        raise NotImplementedError
    
    def _save(self):
        raise NotImplementedError

class RandomDiscretePolicy(BasePolicy):
    def __init__(self, num_actions, rng=None):
        self.num_actions = num_actions
        self.rng = rng
        
    def sample_action(self, state):
        action = self.rng.randint(0, self.num_actions)
        return action, 1./self.num_actions, {}        
    
class RandomGaussianPolicy(BasePolicy):
    def __init__(self, num_actions, rng=None):
        self.num_actions = num_actions
        self.rng = rng
        
    def sample_action(self, state):
        action = self.rng.randn(self.num_actions)
        return action, np.prod(scipy.stats.norm.pdf(action)), {}        
          

class OUPolicy(BasePolicy):
    ''' implements a Ornstein Uhlenbeck process for exploration'''
    def __init__(self, num_actions, mu=0, theta=0.15, sigma=0.3, rng=None, policy=None):
        self.num_actions = num_actions
        self.rng = rng
        self.mu = mu
        self.theta = theta 
        self.sigma = sigma
        self._base_policy = RandomGaussianPolicy(num_actions,rng) if policy is None else policy     
        self.reset() 
        
    def reset(self):
        self.state = np.ones(self.num_actions) * self.mu

    def evolve_state(self, state):
        x = np.copy(self.state)
        dx = self.theta * (self.mu - x) + self.sigma * self.rng.randn(len(x))
        self.state = x + dx
        return state

    def sample_action(self, state):
        action, pa, info = self._base_policy.get_action(state)
        ou_state = self.evolve_state()
        return action + ou_state, pa, info
    

 
          
            
class UniformContinuousPolicy(BasePolicy):
    def __init__(self, low, high, rng=None):
        self._low = low
        self._high = high        
        self._prob = 1.0/np.prod(self._high-self._low)
        self.rng = rng
        
    def sample_action(self, state):
        dim = len(self._high)
        action = self.rng.rand(dim)
        action = action * (self._high-self._low) + self._low
        
        return action, self._prob, {}        
            
class LinearPolicy(BasePolicy):
    def __init__(self, K, sigma, rng=None):
        self._K = K
        self._sigma = sigma
        self.rng = rng
        
    def reset(self):
        pass
        
    def sample_action(self, state):
        mean = np.dot(self._K, state)
        noise = self.rng.randn(len(mean))
        sigma = self._sigma
        
        action = mean+noise*sigma        
        return action, np.prod(scipy.stats.norm.pdf(noise)), {}
        
class SineWavePolicy(BasePolicy):
    def __init__(self, amps, periods, phases):
        self._amps = amps
        self._scales = 2 * np.pi / periods
        self._phases = phases * np.pi / 180.0        
        self._t = 0
        
    def reset(self):
        self._t = 0
        
    def sample_action(self, state):
        a = self._amps * np.sin(self._t * self._scales + self._phases)
        self._t += 1
        return a, 1.0, {}        
            