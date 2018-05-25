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
import gym, gym.error
import gym.monitoring

def test_continuous_model(args, flname):
    np.random.seed(100)
    if args.env=='Swimmer-v1':
        import mujoco_py
        from gym.envs.mujoco import mujoco_env
        env = PartiallyObservableEnvironment(GymEnvironment('Swimmer-v1', discrete = False),\
                                              np.array([0,1,2])) 
    elif args.env=='CartPole-v0':
        env = PartiallyObservableEnvironment( ContinuousEnvironment(args.env, CartpoleContinuousSimulator()),\
                                              np.array([0,2]))
    env.reset()
    start_size = args.initN
    x_dim, output_dim = env.dimensions
    if args.gen=='reward':
        generator = HighRewardTrainSetGenerator(0.3, args.batch, max_size=args.maxtrajs, start_size=start_size) #more peaked in higher rewards
    elif args.gen =='exp':
        generator = ExpTrainSetGenerator(0.2, args.batch, max_size=args.maxtrajs, start_size=start_size) 
    elif args.gen =='boot':
        generator = BootstrapTrainSetGenerator(args.batch, max_size=args.maxtrajs, start_size=start_size)
    elif args.gen == 'last':
        generator = BootstrapTrainSetGenerator(args.numtrajs, max_size=args.numtrajs+args.initN, start_size=start_size)#LastTrainSetGenerator(start_size=start_size)
    else:
        generator = TrainSetGenerator(max_size=args.maxtrajs)#, start_size=start_size)
    params = vars(args)
    #create model with refine PSRs
    start_gen = HighRewardTrainSetGenerator(0.3, args.initN, start_size=start_size)
    model = BatchFilteringRefineModel(batch_gen=generator, file=flname, env=env, params=params, start_gen=start_gen);
    if args.fext=='rff':
        feat_set = create_RFFPCA_featureset(args.Hdim, args.dim, args.kw)
    elif args.fext=='nystrom':
        feat_set = create_NystromPCA_featureset(args.Hdim, args.dim, args.kw)
    elif args.fext=='pca':
        feat_set = create_PCA_featureset(args.Hdim, args.dim)
    
    #collect initial trajectories and update model
    model_init = ObservableModel(x_dim)
    pi_learn = ContinuousPolicy(x_dim, output_dim, 0, [16])
    start_trajs = env.run(model_init, pi_learn, args.initN if args.initN > 0.0 else args.numtrajs, args.len, render=False)
    model.initialize_model(feat_set,start_trajs)
    #assert len(model._batch_gen._trajs)==0, embed()
    #policy learner
    pi_explore = ContinuousPolicy(model.state_dimension, output_dim, args.nL, args.nh) 
    PiUpdator = VR_Reinforce_PolicyUpdater(pi_explore, lr=args.lr)

    #save videos
    if args.monitor<>'':
        env._base.env.monitor.start(flname+args.monitor, seed=387387, force=True) #call env.env if just GymEnvironment
        env._base.env.monitor.configure(video_callable=lambda count: count%int(args.numtrajs)==0) #1 per iteration
#             import gym.wrappers
#             env.env = gym.wrappers.Monitor(env=env.env, directory=flname+args.monitor,\
#                                            force=True, video_callable=lambda count: count%100==0)


    
    #iterate
    learn_model_policy(PiUpdator, model, env, 
                    max_traj_len = args.len, num_trajs = args.numtrajs, 
                    num_iter = args.iter, plot=True)

    if args.datapath<>'':
        end_trajs = env.run(model, pi_explore, args.render, args.len, render=False)
        model.save_model(flname+args.datapath)
        save_data(flname+'results.pkl', [end_trajs])
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
    parser.add_argument('--batch', type=int, default=0, help='max number of training trajectories.')
    parser.add_argument('--dim', type=int, default=40, help='PSR dimension (latente space dim).')
    parser.add_argument('--rff', type=int, default=1000, help='Random Fourier Features dimension.')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate for VR Reinforce')
    parser.add_argument('--len', type=int, default=100, help='Maximum of nominal trajectory length')
    parser.add_argument('--env', type=str, default='CartPole-v0', help='Gym environment')
    parser.add_argument('--rstep', type=float, default=5e-4, help='gradient descent step size')
    parser.add_argument('--tfile', type=str, default='results/', help='Directory containing test data.')
    parser.add_argument('--monitor', type=str, default='', help='monitor file.')
    parser.add_argument('--render', type=int, default=100, help='render final trajectories number.')
    
    #parser.add_argument('--nh', type=int, default='results/', help='number of hidden units.')
    args = parser.parse_args()
    
    test_file = args.tfile+'/%s/d%d_f%d_p%d_rff%d_r%.3f_iter%d_R%d_N%d_B%d_lr%.1e_l%d_rstp%.1e/'%(\
                args.env, args.dim, args.fut, args.past, args.rff, args.reg, args.iter, args.refine,\
                args.numtrajs, args.batch, args.lr, args.len, args.rstep)
    mkpath(test_file)
    print 'Dumping results to:\n%s'%test_file
    test_continuous_model(args, test_file)