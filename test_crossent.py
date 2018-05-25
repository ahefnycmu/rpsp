#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 23:36:26 2017

@author: ahefny
"""

import environments
import models
import policies
import numpy as np
import scipy.stats
import scipy.linalg
from crossent_opt import learn_policy_crossentropy
import matplotlib.pyplot as plt

np.random.seed(0)

class LinearPolicy(policies.BasePolicy):
    def __init__(self, K, sigma):
        self._K = K
        self._sigma = sigma
        
    def reset(self):
        pass
        
    def sample_action(self, state, return_prob = False):
        mean = np.dot(self._K,state.reshape((-1,1))).ravel()
        noise = np.random.randn(len(mean))
        sigma = self._sigma
        
        if return_prob:
            return mean+noise*sigma, np.prod(scipy.stats.norm.pdf(noise))
        else:
            return mean+noise*sigma

#swimmer = gym.make('Swimmer-v1')
#from gym.wrappers import Monitor
#swimmer = Monitor(swimmer, directory='/tmp/gym-swimmer/',video_callable=False, force=True)
#env = environments.GymEnvironment(swimmer, discrete=False)
import dart_environments
env = dart_environments.DartSwimmer()
#env = environments.PartiallyObservableEnvironment(env, np.array(xrange(3)))
d_o,d_a = env.dimensions
model = models.ObservableModel(d_o)            
#model = models.FiniteHistoryModel(d_o,2)
d_s = model.state_dimension
            
def theta2linearpolicy(theta):
    return LinearPolicy(theta.reshape((d_a,d_s)), 0.0)
    
def linearpolicy2theta(policy):
    return policy._K.ravel()
            
def plot(data):
    mu = data['mu']    
    X = data['X']
    Y = data['Y']

    _,_,V = scipy.linalg.svd(X, full_matrices=False)
    V = V.T
    X = np.dot(X,V)
    Y = np.dot(Y,V)
    mu = np.dot(mu,V)
    
    plt.cla()
    plt.scatter(X[:,0],X[:,1],color='y')       
    plt.pause(0.5)        
    plt.scatter(Y[:,0],Y[:,1],color='r')           
    plt.pause(0.5)
    plt.scatter(mu[0],mu[1],color='b',s=40)
    plt.pause(0.1)

def log(data):
    theta = data['mu']
    print '%d: %f' % (data['it'], data['target'])
    #print data['mu']
    #print data['S']

    plot(data)
    mean_policy = theta2linearpolicy(theta)
    env.run(model, mean_policy, 1, 500, render=True)

policy = LinearPolicy(np.random.randn(d_a,d_s), 0.0)
theta0 = np.zeros(d_s*d_a)
S0 = np.ones(d_s*d_a)/(d_s*d_a)

print 'Starting CrossEnt'

learn_policy_crossentropy(theta0, theta2linearpolicy,
                          model, env, S0=S0, max_traj_len=500, num_trajs=2,
                          num_iter=15, sample_size=50, ce_logger=log, ce_alpha=1.0)            
