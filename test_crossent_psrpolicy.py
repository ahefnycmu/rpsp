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
import envs.environments
import models
import policies
import numpy as np
import crossent_opt
import matplotlib.pyplot as plt
import NN_policies
import psr_lite.datagen.gen_swimmers_mujoco as mujoco_swm
import psr_lite.rffpsr
import psr_lite.rffpsr_rnn
import psr_lite.psrlite_policy

np.random.seed(0)

parser = argparse.ArgumentParser(description='Swimmer Robot with PSR model + CEM')
parser.add_argument('--policy', type=str, default='linear')
args, unknows_args = parser.parse_known_args()

policy_string = args.policy

max_traj_length = 500
eval_traj_length = 500        
eval_dir= '/tmp/gym/%s_%s/' % ('psrnet', policy_string)
        
register(
    id='Swimmer-cap-v1',
    entry_point='gym.envs.mujoco:SwimmerEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': eval_traj_length},
    reward_threshold=360.0,
)        

env = environments.GymEnvironment(gym.make('Swimmer-v1'), discrete=False)
env = environments.PartiallyObservableEnvironment(env, np.array(xrange(3)))
monitor = Monitor(gym.make('Swimmer-cap-v1'), directory=eval_dir,
                  force=True, video_callable=lambda it:True, write_upon_reset=True)
eval_env = environments.GymEnvironment(monitor, discrete=False)       
eval_env = environments.PartiallyObservableEnvironment(eval_env, np.array(xrange(3)))
d_o,d_a = env.dimensions
rwd_trace = []

model = models.ObservableModel(d_o)

def evaluate_policy(policy):
    t = eval_env.run(model, policy, 1, eval_traj_length, render=False)    
    rwd_trace.append(np.sum(t[0].rewards))
    np.savetxt(eval_dir+'rwd.csv', np.array(rwd_trace))  
    
    plt.cla()
    plt.plot(np.array(rwd_trace))
    plt.savefig(eval_dir+'rwd.png')
    
# Exploration Trajectories
print 'Generating exploration trajectories'
pi_exp = policies.UniformContinuousPolicy(low=np.array([-1,-1]),
                                              high=np.array([1,1]))
exp_trajs = env.run(model, pi_exp, 10, max_traj_length)
X_obs_rnd = [c.obs for c in exp_trajs]
X_act_rnd = [c.act for c in exp_trajs]
X_obs_good,X_act_good = mujoco_swm.gen_trajs(10, max_traj_length, policy='full') # Can use policy='full'/'partial'

X_obs = X_obs_good+X_obs_rnd
X_act = X_act_good+X_act_rnd

print 'Training PSR'
feats = psr_lite.feat_extractor.create_RFFPCA_featureset(1000,50)
psr = psr_lite.rffpsr.RFFPSR(10, 10, 50, feature_set=feats)
psr.train(X_obs, X_act)

print 'Building policy network'
psrrnn = psr_lite.rffpsr_rnn.RFFPSR_RNN(psr)
if policy_string == 'linear':
    pi_react = NN_policies.ContinuousPolicy(x_dim = psr.state_dim, output_dim = d_a, num_layers = 0, nh = [10]);
elif policy_string == 'nn':
    pi_react = NN_policies.ContinuousPolicy(x_dim = psr.state_dim, output_dim = d_a, num_layers = 1, nh = [10]);
else:
    pi_react = None
pi = psr_lite.psrlite_policy.RFFPSRNetworkPolicy(psrrnn, pi_react, np.array([0,0]))

def theta2policy(theta):
    pi.set_params(theta)
    return pi
               
# Initial evaluation
evaluate_policy(pi)

# Training
def log(data):
    it = data['it']+1
    print 'Finished iteration %d' % (it)
    crossent_opt.celog_plotsample(data)
    plt.savefig('%sce_%0.5d.png' % (eval_dir,it))
    
    theta = data['mu']    
    mean_policy = theta2policy(theta)
    evaluate_policy(mean_policy)

beta_rwd = 1.0
beta_pred = 1.0
    
def cost(policy, trajs):
    return beta_rwd * crossent_opt.neg_rwd(policy, trajs) \
    + beta_pred * np.mean([policy._psrnet.traj_1smse(t.obs, t.act) for t in trajs])
    
theta0 = pi.get_params()
S0 = np.ones_like(theta0) #/(d_f*d_a)

crossent_opt.learn_policy_crossentropy(theta0, theta2policy, model, env,
                                       S0=S0, max_traj_len=max_traj_length, num_trajs=2,
                                       num_iter=100, sample_size=50, ce_logger=log,
                                       ce_alpha=1.0, trajs_cost=cost) 
    
env.close()
eval_env.close()

