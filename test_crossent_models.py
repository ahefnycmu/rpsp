#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 23:36:26 2017

@author: ahefny,zmarinho
"""

import gym

import envs.environments
import models
import policies
import numpy as np
import scipy.stats
import scipy.linalg
import crossent_opt
from crossent_opt import learn_policy_crossentropy
import matplotlib.pyplot as plt
from envs.load_environments import *
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

def test_crossentropy(args, flname): 
    args.flname = flname
    env, model_exp, pi_exp = env_dict[args.env](args)
    env = NoisyEnvironment(env, args.obsnoise)
    env.reset()
    
    d_o,d_a = env.dimensions
    model = models.ObservableModel(d_o)            
    #model = models.FiniteHistoryModel(d_o,2)
    d_s = model.state_dimension
    results = {'rewards':[],'mse':[], 'exp':args.flname}
   
    if model_exp is None:
        model_exp = ObservableModel(d_o)
    if pi_exp is None:
        pi_exp = policies.RandomGaussianPolicy(d_o)
    print 'dimension:', d_o
        
    print 'build updater ... ',
    ''' run the observable model with reactive policy'''

    if args.method=='lite-obsCE' or args.method=='lite-contCE':
        ''' run the psr network with obs model or psr model'''
        model = model_exp
        rwd_trace = []
        def evaluate_policy(policy):
            trajs = env.run(model, policy, 1, args.len, render=False)    
            rwd_trace.append(np.sum(trajs[0].rewards))
            np.savetxt(args.flname+'rwd.csv', np.array(rwd_trace))  
            
            plt.cla()
            plt.plot(np.array(rwd_trace))
            plt.savefig(args.flname+'rwd.png')
            results['rewards'].append([np.sum(t.rewards) for t in trajs])
        # Exploration Trajectories
        print 'Generating exploration trajectories'
        pi_exp = policies.RandomGaussianPolicy(d_a)
        pi_exp = policies.UniformContinuousPolicy(low=np.array([-1,-1]),
                                              high=np.array([1,1]))
        #pi_exp = ContinuousPolicy(x_dim, output_dim, 0, [16])
        exp_trajs = env.run(model_exp, pi_exp, args.initN, args.len)
        col_trajs = [(t.obs, t.act) for t in exp_trajs]
        X_obs_rnd = [c.obs for c in exp_trajs]
        X_act_rnd = [c.act for c in exp_trajs]
        try:
            X_obs_good,X_act_good = mujoco_swm.gen_trajs(args.initN, args.len, policy='full') # Can use policy='full'/'partial'
        except Exception:
            X_obs_good =np.zeros(X_obs_rnd.shape); X_act_good=np.zeros(X_act_rnd.shape);
        X_obs = X_obs_rnd + X_obs_good
        X_act = X_act_rnd + X_act_good    
        rffpsr_model, rnnmodel = model_call(args, data=[X_obs,X_act], x_dim=d_o)
        pi_react = NN_policies.ContinuousPolicy(x_dim = rffpsr_model.state_dimension, output_dim = d_a, num_layers = args.nL, nh = args.nh);    
        pi = psr_lite.psrlite_policy.RFFPSRNetworkPolicy(rnnmodel, pi_react, np.zeros((d_a)))
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
            plt.savefig('%sce_%0.5d.png' % (args.flname,it))
            
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
        
        pi = crossent_opt.learn_policy_crossentropy(theta0, theta2policy, model, env,
                                               S0=S0, max_traj_len=args.len, num_trajs=2,
                                               num_iter=args.iter, sample_size=args.numtrajs, ce_logger=log,
                                               ce_alpha=1.0, trajs_cost=cost)
        env.close()
    else:
        if args.method=='obsCE':
            model = model_exp
        elif args.method=='arCE':
            model = FiniteHistoryModel(obs_dim=d_o, past_window=args.past)
        elif args.method=='rnn':
            raise NotImplementedError
        elif args.method=='lstmCE':
            model = model_exp
            pi_react = NN_policies.RNN_Continuous_Policy(x_dim=model.state_dimension, a_dim = None, output_dim=d_a,nh = args.nh[0], LSTM = True) #nh=64
            PiUpdator = NN_policies.VR_Reinforce_RNN_PolicyUpdater(policy = pi_react, max_traj_length = args.len, num_trajs = args.numtrajs, lr = args.lr)
        elif args.method=='psr_cont':
            raise NotImplementedError
        
        def theta2linearpolicy(theta, ):
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
            plt.scatter(np.arange(X.shape[0]),X[:,0], color= 'y')
            plt.scatter(np.arange(Y.shape[0]),Y[:,0], color= 'r')
            plt.axhline( y=mu[0], linewidth=2, color = 'b')
            
            #plt.scatter(X[:,0],X[:,1],color='y')       
            #plt.pause(0.5)        
            #plt.scatter(Y[:,0],Y[:,1],color='r')           
            #plt.pause(0.5)
            #plt.scatter(mu[0],mu[1],color='b',s=40)
            #plt.pause(0.1)
        
        def log(data):
            theta = data['mu']
            print '%d: %f' % (data['it'], data['target'])
            #print data['mu']
            #print data['S']
            plot(data)
            mean_policy = theta2linearpolicy(theta)
            trajs = env.run(model, mean_policy, 1, 500, render=True)
            results['rewards'].append([np.sum(t.rewards) for t in trajs])
            return 
        
        policy = LinearPolicy(np.random.randn(d_a,d_s), 0.0)
        theta0 = np.zeros(d_s*d_a)
        S0 = np.ones(d_s*d_a)/(d_s*d_a)
        
        print 'Starting CrossEnt'
        
        pi = crossent_opt.learn_policy_crossentropy(theta0, theta2linearpolicy,
                                  model, env, S0=S0, max_traj_len=args.len, num_trajs=2,
                                  num_iter=args.iter, sample_size=args.numtrajs, ce_logger=log, ce_alpha=1.0)   
    print 'done'
    print 'len:',args.len, 'num trajs:', args.numtrajs, 'iter:',args.iter
     
    results['K']= pi._K
    #results['pi_params'] = pi.get_params() 
    return results