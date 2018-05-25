#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 14:49:01 2016

@author: ahefny
"""

import scipy.linalg
import numpy as np
import matplotlib.pyplot as plt
from policy_learn import BasePolicyUpdater
             
'''
Cross entropy optimizer with diagonal covariance Gaussian sampling distribution
See: https://people.smp.uq.edu.au/DirkKroese/ps/aortut.pdf

Parameters:
d - Dimension of variable to optimize
cost_function - Handle to function or callable object to compute f(x) = cost at x
mu0, S0 - Vectors of initial sampling mean and standard deviation for each dimension
alpha - Controls the aggressiveness of updates: mu_{t+1} = alpha mu_{new} + (1-alpha) mu_t
noise - Additional standard deviation added to the estimate of S 
sample_size - Number of samples per iteration
num_iterations - Number of iterations
percentile (between 0 and 100) - Determines the percentile of costs in the current
     iteration to set as a target for the next iteration.         
'''
def gaussian_crossent_opt(d, cost_function, mu0=None, S0=None, alpha=0.1, noise=0.0, sample_size=100,
                          num_iterations=1000, percentile=5, logger=None, return_distribution=False):
    if mu0 is None: mu0 = np.zeros(d)        
    if S0 is None: S0 = np.ones(d)
    
    mu = mu0
    S = S0
    
    p_idx = percentile * sample_size // 100
    
    X = None
    idx = None
    best_X = None
    best_cost = float('inf')
    
    for it in xrange(num_iterations):
        # TODO: Add termination condition
        X = np.random.randn(sample_size, d) * S + mu        

        costs = np.array([cost_function(X[i,:]) for i in xrange(sample_size)])        
        idx = np.argsort(costs)

        if costs[idx[0]] < best_cost:
            best_X = X[idx[0],:]
            best_cost = costs[idx[0]]
            
        target = costs[idx[p_idx]]
                
        mask = np.array(costs) <= target
        Y = X[mask,:]
        mu_new = np.mean(Y, 0)
        S_new = np.sqrt(np.mean(Y*Y, 0) - (mu_new * mu_new)) + noise
        
        mu = mu_new * alpha + mu * (1-alpha)
        S = S_new * alpha + S * (1-alpha)
                
        if logger is not None:
            logger({'X':X, 'Y':Y, 'mu':mu, 'S':S, 'target':target, 'it':it})
            
    if return_distribution:
        return best_X, mu, S
    else:
        return best_X

def neg_rwd(policy, trajs):
    return sum(np.sum(-t.rewards) for t in trajs) / len(trajs)
        
def policy_cost(policy, environment, model, num_trajs, max_traj_len, trajs_cost): 
    trajs = environment.run(model, policy, num_trajs, max_traj_len)        
    cost = trajs_cost(policy, trajs)
    return cost
            
def learn_policy_crossentropy(theta0, theta2policy, model, environment, S0=None,
                              max_traj_len=100, num_trajs=10, num_iter=100,
                              sample_size=100, ce_alpha=0.1, ce_logger=None, trajs_cost=neg_rwd):            
    cost = lambda theta: policy_cost(theta2policy(theta), environment, model,
                                     num_trajs, max_traj_len, trajs_cost)
    theta_opt = gaussian_crossent_opt(theta0.size, cost, mu0=theta0, S0=S0, sample_size=sample_size,
                                       num_iterations=num_iter, alpha=ce_alpha, logger=ce_logger)
    
    return theta2policy(theta_opt)
 
def celog_plotsample(data, pause_time=0.01):
    mu = data['mu']    
    X = data['X']
    Y = data['Y']

    X = X - np.mean(X, axis=0)

    _,_,V = scipy.linalg.svd(X, full_matrices=False)
    V = V.T
    X = np.dot(X,V)
    Y = np.dot(Y,V)
    mu = np.dot(mu,V)
    
    plt.cla()
    plt.scatter(X[:,0],X[:,1],color='y')  
    plt.hold(True)     
    plt.pause(pause_time)        
    plt.scatter(Y[:,0],Y[:,1],color='r')           
    plt.pause(pause_time)
    plt.scatter(mu[0],mu[1],color='b',s=40)
    plt.pause(pause_time)
    plt.hold(False)     
           
#'''
#A PolicyUpdater that uses cross entropy method. To do the update, this updater
#samples trajectories using environment and model rather than using 
#the batch of trajectories supplied to the update method.
#'''
#class CrossEntropyPolicyUpdater(BasePolicyUpdater):
#    def __init__(self, environment, model, theta0, theta2policy, S0=None,
#                 num_eval_trajs=100, max_eval_traj_len=100, ce_iterations=100):
#        self._theta_ = theta0
#        self._policy = theta2policy(theta0)     
#        self._theta2policy = theta2policy
#        
#        self._mu = theta0                        
#        self._cost = lambda theta: _policy_cost(theta2policy(theta), environment, 
#                                                model, num_eval_trajs, max_eval_traj_len)
#        
#        self._ce_iterations = ce_iterations
#        
#    def update(self, trajs):        
#        self._theta = _gaussian_crossent_opt(self._mu.size, self._cost, self._theta,
#                                             num_iterations=self._ce_iterations)        
#        self._policy = self._theta2policy(self._theta)        
#                
#    @property
#    def policy(self):
#        return self._best_policy
                
if __name__ == '__main__':
    def f(data):
        mu = data['mu']
        print mu, data['S'], data['target']

        X = data['X']
        Y = data['Y']

        plt.cla()
        plt.scatter(X[:,0],X[:,1],color='y')   
        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.pause(0.5)        
        plt.scatter(Y[:,0],Y[:,1],color='r')           
        plt.pause(0.5)
        plt.scatter(mu[0],mu[1],color='b',s=40)
        plt.pause(0.5)
                
    
    cost = lambda x: (x[0]-1)*(x[0]-1) + x[1]*x[1] # Minimum is (1,0)
    #cost = lambda x: np.sin(x[0]) + np.cos(x[1])
    mu0 = np.array([-1,1])    
    
    x_opt = _gaussian_crossent_opt(2,cost,mu0, num_iterations=100, logger=f)
    print x_opt
    
    
    


        
        
        
                 
    
    