#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 00:26:33 2017

@author: ahefny
"""

import argparse
import gym
from gym.envs.registration import register
from gym.wrappers import Monitor
import environments
import models
import policies
import numpy as np
import scipy.stats
import scipy.linalg
import crossent_opt
import matplotlib.pyplot as plt
import psr_models.rff_psr_model
import psr_models.features.feat_extractor as psr_feats
import PSIM.psim_model
import PSIM.kernel_ridge_regression
import NN_policies

np.random.seed(0)

parser = argparse.ArgumentParser(description='Swimmer Robot with PSR model + CEM')
parser.add_argument('--model', type=str, default='obs')
parser.add_argument('--policy', type=str, default='linear')
args, unknows_args = parser.parse_known_args()

model_string = args.model
policy_string = args.policy

class LinearPolicy(policies.BasePolicy):
    def __init__(self, K, sigma):
        self._K = K
        self._sigma = sigma
        self.discrete = False
        
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
            
    def compute_action_prob(self,s,a):
        noise = (a-np.dot(self._K,s))/self._sigma
        return np.prod(scipy.stats.norm.pdf(noise))

eval_traj_length = 500        
eval_dir= '/tmp/gym/%s_%s/' % (model_string, policy_string)
        
register(
    id='Swimmer-cap-v1',
    entry_point='gym.envs.mujoco:SwimmerEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': eval_traj_length},
    reward_threshold=360.0,
)        

env = environments.GymEnvironment(gym.make('Swimmer-v1'), discrete=False)
#env = environments.PartiallyObservableEnvironment(env, np.array(xrange(4)))
monitor = Monitor(gym.make('Swimmer-cap-v1'), directory=eval_dir,
                  force=True, video_callable=lambda it:True, write_upon_reset=True)
eval_env = environments.GymEnvironment(monitor, discrete=False)       
#eval_env = environments.PartiallyObservableEnvironment(eval_env, np.array(xrange(4)))
d_o,d_a = env.dimensions
rwd_trace = []

def evaluate_policy(model, policy):
    t = eval_env.run(model, policy, 1, eval_traj_length, render=False)    
    rwd_trace.append(np.sum(t[0].rewards))
    np.savetxt(eval_dir+'rwd.csv', np.array(rwd_trace))  
    
    plt.cla()
    plt.plot(np.array(rwd_trace))
    plt.savefig(eval_dir+'rwd.png')
    
# Exploration Trajectories
print 'Generating exploration trajectories'
exp_policy = policies.UniformContinuousPolicy(low=np.array([-1,-1]),
                                              high=np.array([1,1]))
exp_model = models.ObservableModel(d_o)
exp_trajs = env.run(exp_model, exp_policy, 50, 500)

# Construct model using exploration trajectories    
print 'Building model'
if model_string == 'obs':
    model = models.ObservableModel(d_o)    
    d_f = d_o    
    theta0_mdl = np.array([])
    theta2mdl = lambda theta: 0    
elif model_string == 'psr':
    psr = psr_models.rff_psr_model.BatchFilteringRefineModel(batch_gen=models.BootstrapTrainSetGenerator(), n_iter_refine=5, file=eval_dir)
    #feats = psr_feats.create_RFFPCA_featureset(1000, 20)
    feats = psr_feats.create_NystromPCA_featureset(1000, 80, 70)
    psr.initialize_model(feats, past=5,fut=5)
    psr.update(exp_trajs)
    
    model = psr
    d_f = psr.state_dimension
    theta0_mdl = psr._get_parameters()
    theta2mdl = lambda theta: psr._set_parameters(theta)
        
d_mdl = len(theta0_mdl)    

if policy_string == 'linear':
    policy = LinearPolicy(np.random.randn(d_a,d_f), 0.0)
    theta2policy = lambda theta: LinearPolicy(theta.reshape((d_a,d_f)), 0.0) 
    d_policy = d_f*d_a
elif policy_string == 'nn':
    policy = NN_policies.ContinuousPolicy(d_f, d_a, 1, [20])
    def theta2policy_nn(theta):
        policy.set_params(theta)
        return policy
        
    theta2policy = theta2policy_nn
    d_policy = len(policy.get_params())
            
def theta_update(theta):
    #theta2mdl(theta[:d_mdl])
    return theta2policy(theta[d_mdl:])
    
# Initial evaluation
evaluate_policy(model, policy)

# Training
def log(data):
    it = data['it']+1
    print 'Finished iteration %d' % (it)
    crossent_opt.celog_plotsample(data)
    plt.savefig('%sce_%0.5d.png' % (eval_dir,it))
    
    theta = data['mu']    
    mean_policy = theta_update(theta)
    evaluate_policy(model, mean_policy)

    np.savetxt('%spolicy_%0.5d.csv' % (eval_dir, it), mean_policy._K, delimiter=',')
    print mean_policy._K

theta0 = np.hstack((theta0_mdl, np.zeros(d_policy)))
S0 = np.ones(d_mdl+d_policy) #/(d_f*d_a)

crossent_opt.learn_policy_crossentropy(theta0, theta_update, model, env,
                                       S0=S0, max_traj_len=500, num_trajs=2,
                                       num_iter=15, sample_size=50, ce_logger=log,
                                       ce_alpha=1.0) 
    
env.close()
eval_env.close()

#psr = psr_models.rff_psr_model.BatchFilteringRefineModel(batch_gen=models.BootstrapTrainSetGenerator())
#raw_trajs = env.run(model, policy, 5, 100)
#feats = psr_feats.create_RFFPCA_featureset(1000, 20)
#psr.initialize_model(feats, past=5,fut=5)
#psr.update(raw_trajs, 1)
#
##psim_learner = PSIM.kernel_ridge_regression.RFF_RidgeRegression(Ridge = 1e-6, bwscale = 1.)
##psim = PSIM.psim_model.PSIM_FilteringModel(psim_learner, d_o, d_a, discrete=False)
#d_f = psr.state_dimension
#policy = LinearPolicy(np.random.randn(d_a,d_f), 1.0)
##psim.set_policy(policy)
#
#print 'Generating Initial Trajs'
#trajs = env.run(psr, policy, 50, 200, render=False)
#
#print 'Updating model'
#psr.update(trajs, 2)
#
#def theta2linearpolicy(theta):
#    return LinearPolicy(theta.reshape((d_a,d_f)), 0.0)
#    
#def linearpolicy2theta(policy):
#    return policy._K.ravel()
#            
#def plot(data):
#    mu = data['mu']    
#    X = data['X']
#    Y = data['Y']
#
#    _,_,V = scipy.linalg.svd(X, full_matrices=False)
#    V = V.T
#    X = np.dot(X,V)
#    Y = np.dot(Y,V)
#    mu = np.dot(mu,V)
#    
#    plt.cla()
#    plt.scatter(X[:,0],X[:,1],color='y')       
#    plt.pause(0.5)        
#    plt.scatter(Y[:,0],Y[:,1],color='r')           
#    plt.pause(0.5)
#    plt.scatter(mu[0],mu[1],color='b',s=40)
#    plt.pause(0.1)
#
#def log(data):
#    theta = data['mu']
#    print '%d: %f' % (data['it'], data['target'])
#    #print data['mu']
#    #print data['S']
#
#    plot(data)
#    mean_policy = theta2linearpolicy(theta)
#    env.run(psr, mean_policy, 1, 500, render=True)
#
#theta0 = np.zeros(d_f*d_a)
#S0 = np.ones(d_f*d_a) #/(d_f*d_a)
#
#learn_policy_crossentropy(theta0, theta2linearpolicy,
#                          psr, env, S0=S0, max_traj_len=500, num_trajs=2,
#                          num_iter=15, sample_size=50, ce_logger=log, ce_alpha=1.0) 



