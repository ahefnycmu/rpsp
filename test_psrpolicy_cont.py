#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 12:38:44 2017

@author: ahefny
"""
import numpy as np
import NN_policies
from environments import *
from simulators import CartpoleContinuousSimulator
import psr_lite.rffpsr
import psr_lite.feat_extractor
import psr_lite.rffpsr_rnn
import psr_lite.psrlite_policy
from models import ObservableModel
import policies
import psr_lite.psr_lite_vrpg
from policy_learn import learn_policy
#import datagen.gen_swimmers_mujoco as mujoco_swm
from IPython import  embed
#import psr_lite.datagen.gen_swimmers_mujoco as mujoco_swm
from psr_lite.utils.plot import call_plot
                    
np.random.seed(200);
#env = GymEnvironment('InvertedPendulum-v1', discrete=False);
#env = partial_obs_Gym_CartPole_Env('CartPole-v0');
#env = GymEnvironment('Acrobot-v1');
#env = GymEnvironment('Swimmer-v1', discrete = False);
#env = PartiallyObservableEnvironment(GymEnvironment('Swimmer-v1', discrete = False), np.array([0,1,2]))
env = PartiallyObservableEnvironment(ContinuousEnvironment('CartPole-v0', CartpoleContinuousSimulator()),np.array([0,2]));
#env = PartiallyObservableEnvironment(GymEnvironment('Swimmer-v1', discrete = False), np.array([0,1,2]))
#env = PartiallyObservableEnvironment(GymEnvironment('CartPole-v0'),np.array([0,2]));

env.reset();
(x_dim, a_dim) = env.dimensions;
print x_dim, a_dim;
output_dim = a_dim
print output_dim
#this model could be replace by PSR or PSIM later.
model = ObservableModel(obs_dim = x_dim); 

max_traj_length = 500
min_traj_length = 6
num_trajs = 20

lambda_psr = {'s1a':1e-3, 's1b':1e-3, 's1c':1e-3, 's1div':1e-3,
            's2ex':1e-5, 's2oo':1e-5, 'filter':1e-3, 'pred':1e-3}

pi_exp = policies.RandomGaussianPolicy(output_dim)
exp_trajs = env.run(model, pi_exp, 10, max_traj_length)
X_obs_rnd = [c.obs for c in exp_trajs]
X_act_rnd = [c.act for c in exp_trajs]
#X_obs_good,X_act_good = mujoco_swm.gen_trajs(10, max_traj_length, policy='full') # Can use policy='full'/'partial'
X_obs = X_obs_rnd #+ X_obs_good
X_act = X_act_rnd #+ X_act_good
  
feats = psr_lite.feat_extractor.create_uniform_featureset(lambda: psr_lite.feat_extractor.FeatureExtractor())
feats = psr_lite.feat_extractor.create_RFFPCA_featureset(1000,50)
psr = psr_lite.rffpsr.RFFPSR(1, 1, 50, feature_set=feats, l2_lambda=lambda_psr)
psr.train(X_obs, X_act)
psrrnn = psr_lite.rffpsr_rnn.RFFPSR_RNN(psr,optimizer_iterations=0)
psrrnn.train(X_obs,X_act)

#pi = NN_policies.ContinuousPolicy(x_dim = x_dim, output_dim = output_dim, num_layers = 0, nh = [10]);
pi_react = NN_policies.ContinuousPolicy(x_dim = psr.state_dimension, output_dim = output_dim, num_layers = 1, nh = [10]);
pi = psr_lite.psrlite_policy.RFFPSRNetworkPolicy(psrrnn, pi_react, np.zeros(output_dim))

pp = call_plot()
def logger(i,trajs, res):
    m = np.mean([np.sum(t.rewards) for t in trajs]); 
    s = np.std([np.sum(t.rewards) for t in trajs]);
    R = [psrrnn.traj_1smse(t.obs, t.act) for t in trajs]
    print "\t\t\tMSE={:3.4f} (std={:3.4f})\tGerror={:3.4f} (std={:3.4f})\tcost={:3.4f} (std={:3.4f})".format( np.mean(R), np.std(R),res[2],res[3],res[0],res[1])
    pp.plot(res[0],res[1],res[2],res[3],m,s)
    tpred = psrrnn.traj_predict_1s(trajs[0].obs, trajs[0].act)
    pp.plot_traj(trajs[0], tpred)

#pi = RNN_Discrete_Policy(x_dim = x_dim, a_dim = a_dim, output_dim=output_dim, nh = 16, LSTM = True);
#pi = NN_policies.RNN_Continuous_Policy(x_dim=x_dim, a_dim = a_dim, output_dim=output_dim,nh = 64, LSTM = True);
#pi = ContinuousPolicy(x_dim = x_dim, output_dim = output_dim, num_layers = 1, nh = [64]);
print 'build updater ... ',
#PiUpdator = NN_policies.VR_Reinforce_PolicyUpdater(policy = pi, lr = 1e-2);
PiUpdator = psr_lite.psr_lite_vrpg.VR_Reinforce_RNN_PolicyUpdater_gcheck(policy = pi, 
                        max_traj_length = max_traj_length,
                        num_trajs = num_trajs, lr = 1e-2, beta_reinf=1.0, beta_pred=1.0,
                        beta_pred_decay=1.0);
print 'done'

learn_policy(PiUpdator, model, env, num_trajs=num_trajs, 
             max_traj_len=max_traj_length, min_traj_length=min_traj_length,
             num_iter = 100, logger=logger);
            
            