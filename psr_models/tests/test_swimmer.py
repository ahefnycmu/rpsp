'''
Created on Feb 01, 2017

@author: zmarinho
'''

import matplotlib
matplotlib.use('Agg') 

import numpy as np
import sys, time, os, math
from IPython import embed
from environments import *
from NN_policies import *
from models import *
from policy_learn import learn_policy, learn_model_policy
from psr_models.features.rff_features import RFF_features
from itertools import imap, chain
from psr_models.rff_psr_model import *   
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
from distutils.dir_util import mkpath
from psr_models.features.feat_extractor import *
import cPickle as pickle
import psr_models.utils.plot_utils as pu
from psr_models.utils.utils import save_data
import mujoco_py
import gym
from gym.envs.mujoco import mujoco_env
import gym.monitoring

def test_continuous_model(args, flname):
    np.random.seed(100)
    #env = ContinuousEnvironment(environment, cartpole_continuous)  #x,theta, x_dot,theta_dot
    env = GymEnvironment(args.env, discrete=False) 
    env.reset()
    if args.monitor<>'':
        #mon = gym.monitoring.Monitor(env, flname+args.monitor)
        env.env.monitor.start(flname+args.monitor)
        env.env.monitor.configure(video_callable=lambda count: count%100==0) 
    
    x_dim, output_dim = env.dimensions
    #always train with batch trajectories more recent trajectories
    #generator = ExpTrainSetGenerator(coeff, batch) 
    generator = BootstrapTrainSetGenerator()
    #create model with refine PSRs
    model = BatchFilteringRefineModel(batch_gen=generator,val_ratio = 0.9, \
                                      reg=args.reg, rstep=args.rstep, n_iter_refine=args.refine,\
                                      file=flname);
    feat_set = create_RFFPCA_featureset(args.rff, args.dim)
    model.initialize_model(feat_set, p=args.dim, past=args.past, fut=args.fut)
    
   
    #collect initial trajectories and update model
    model_init = ObservableModel(x_dim)
    pi_learn = ContinuousPolicy(x_dim, output_dim, 0, [16])
    start_trajs = env.run(model_init, pi_learn, args.numtrajs, args.len, render=False)
    model.update(start_trajs)
    
    #policy learner
    pi_explore = ContinuousPolicy(model._filtering_model.dim, output_dim, 0, [16]) 
#    PiUpdator = VR_Reinforce_RNN_PolicyUpdater(
#                             policy = pi_explore, 
#                             max_traj_length = max_traj_length,
#                             num_trajs = args.numtrajs);
    PiUpdator = VR_Reinforce_PolicyUpdater(pi_explore, lr=args.lr)

    #iterate
    results = learn_model_policy(PiUpdator, model, env, 
                    max_traj_len = args.len, num_trajs = args.numtrajs, 
                    num_iter = args.iter)

    #plot data
    pu.plot_boxes(model._batch_gen._trajs.rewards, batch=args.numtrajs, filename=flname)
    pu.plot_results(results, args.numtrajs, filename=flname)
    if args.render:
        #check env
        #start_trajs = env.run(model_init, pi_learn, args.numtrajs, args.len, render=True)
        end_trajs = env.run(model, pi_explore, 10, args.len, render=True)
    if args.datapath<>'':
        model.save_model(flname+args.datapath)
        save_data(flname+'results.pkl', [results, end_trajs])
    embed()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Swimmer Robot with PSR model + VR Reinforce')
    parser.add_argument('--datapath', type=str, default='', \
                        help='Directory containing trained_models. Each pkl file contains matrices W2ext W2oo W2fut.')
    parser.add_argument('--fut', type=int, default=5, help='future window')
    parser.add_argument('--past', type=int, default=7, help='past window')
    parser.add_argument('--reg', type=float, default=1e-3, help='model regularization constant')
    parser.add_argument('--iter', type=int, default=10, help='number of training iterations')
    parser.add_argument('--refine', type=int, default=5, help='number of model refinment iterations')
    parser.add_argument('--numtrajs', type=int, default=100, help='number of trajectories per iterations')
    parser.add_argument('--batch', type=int, default=1200, help='max number of training trajectories.')
    parser.add_argument('--dim', type=int, default=40, help='PSR dimension (latente space dim).')
    parser.add_argument('--rff', type=int, default=1000, help='Random Fourier Features dimension.')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate for VR Reinforce')
    parser.add_argument('--len', type=int, default=100, help='Maximum of nominal trajectory length')
    parser.add_argument('--env', type=str, default='CartPole-v0', help='Gym environment')
    parser.add_argument('--rstep', type=float, default=5e-4, help='gradient descent step size')
    parser.add_argument('--tfile', type=str, default='results/', help='Directory containing test data.')
    parser.add_argument('--monitor', type=str, default='', help='monitor file.')
    #parser.add_argument('--nh', type=int, default='results/', help='number of hidden units.')
    args = parser.parse_args()
    
    test_file = args.tfile+'/%s/d%d_f%d_p%d_rff%d_r%.3f_I%d_R%d_N%d_B%d_lr%.1e_l%d_rstp%.1e/'%(\
                args.env, args.dim, args.fut, args.past, args.rff, args.reg, args.iter, args.refine,\
                args.numtrajs, args.batch, args.lr, args.len, args.rstep)
    mkpath(test_file)
    print 'Dumping results to:\n%s'%test_file
    test_continuous_model(args, test_file)