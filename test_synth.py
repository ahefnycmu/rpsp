#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 14:00:28 2017

@author: ahefny
"""

import numpy as np

from models import ObservableModel
import envs.synth
from NN_policies import ContinuousPolicy, RNN_Continuous_Policy, VR_Reinforce_PolicyUpdater, VR_Reinforce_RNN_PolicyUpdater
from policy_learn import learn_policy
import nn_policy_updaters

rng = np.random.RandomState(300)
env = envs.synth.ReferenceValEnvironment(rng)

(x_dim, a_dim) = env.dimensions;
print x_dim, a_dim
output_dim = a_dim
print output_dim
#this model could be replace by PSR or PSIM later.
model = ObservableModel(obs_dim = x_dim); 

max_traj_length = 50;
num_trajs = 5;

pi = ContinuousPolicy(x_dim = x_dim, output_dim = output_dim, num_layers = 0, nh = [], activation='relu', rng=rng);
#pi = RNN_Continuous_Policy(x_dim=x_dim, a_dim = a_dim, output_dim=output_dim, nh = 1, LSTM = True, rng=rng)

def logger(x,trajs,res):
    pass
    #import pdb; pdb.set_trace()
    for k,v in res.items():            
        print k,v 
        #for p in pi.params:
    #    print p.name, p.get_value()
    
print 'build updater'
#PiUpdator = VR_Reinforce_PolicyUpdater(pi,max_traj_length, num_trajs, lr = 1e-2, gamma = 0.98, baseline = False)

#PiUpdator = VR_Reinforce_RNN_PolicyUpdater(policy = pi, 
#        max_traj_length = max_traj_length,
#        num_trajs = num_trajs, lr = 1e-2, gamma = 0.98, baseline = False)


#PiUpdator = nn_policy_updaters.VRPGPolicyUpdater(pi,
#                                                 max_traj_length=max_traj_length, 
#                                                 gamma=0.98,
#                                                 num_trajs=num_trajs, 
#                                                 baseline='None', #nn_policy_updaters.ZeroBaseline(),
#                                                 #lr=1e-2, clips=[-1,1], ext=False)                                            
#                                                 lr=1e-2, normalize_grad=True)                                            

#PiUpdator = nn_policy_updaters.VRPGPolicyUpdater(pi,
#                                                 max_traj_length=max_traj_length, 
#                                                 gamma=0.98,
#                                                 num_trajs=num_trajs, 
#                                                 baseline='None', #nn_policy_updaters.ZeroBaseline(),
#                                                 lr=1e-2)                                            

PiUpdator = nn_policy_updaters.TRPOPolicyUpdater(pi,
                                                 max_traj_length=max_traj_length, gamma=0.98,
                                                 num_trajs=num_trajs, baseline=nn_policy_updaters.LinearBaseline(),
                                                 lr=5e-3)                                            

#PiUpdator = nn_policy_updaters.RNN_TRPOPolicyUpdater(pi,
#                                                 max_traj_length=max_traj_length, gamma=0.98,
#                                                 num_trajs=num_trajs, baseline=nn_policy_updaters.LinearBaseline(),
#                                                 lr=1e-2)                                            
                                    
#PiUpdator = VR_Reinforce_RNN_PolicyUpdater(policy = pi, 
#        max_traj_length = max_traj_length,
#        num_trajs = num_trajs, lr = 1e-2, gamma = 0.98, baseline = False)
      

      
learn_policy(PiUpdator, model, env,
            max_traj_len = max_traj_length, num_trajs = num_trajs, 
            num_iter = 5000, logger=logger);
    
