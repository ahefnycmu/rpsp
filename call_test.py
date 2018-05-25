# -*- coding: utf-8 -*-
'''
Created on Tue Feb 7 12:16:30 2017

@author: zmarinho
'''

import globalconfig
import psr_lite.globalconfig
import argparse
import numpy as np
from distutils.dir_util import mkpath
import json
from IPython import embed
from stats_test import run_Nmodel, run_model

from psr_lite.psr_lite.utils.log import Logger

class LoadFromJSONFile (argparse.Action):
    def __call__(self, parser, namespace, values, option_string = None):
        with values as f:
            data = json.loads(f.read())
            
            for (k,v) in data.items():                
                setattr(namespace, k, v)
                        
            setattr(namespace, 'original_tfile', data['tfile'])
            setattr(namespace, 'tfile', '')

def add_boolean_option (parser, arg_name, default=False, false_name=None, help=help, help_false=None):
    if false_name is None:
        false_name = 'no_' + arg_name
        
    parser.add_argument('--' + arg_name, dest=arg_name, action='store_true', help=help)
    parser.add_argument('--' + false_name, dest=arg_name, action='store_false', help=help_false)
    parser.set_defaults(**{arg_name : default})
    
def get_parser():
    parser = argparse.ArgumentParser(description='Swimmer Robot with PSR model + VR Reinforce')
    
    parser.add_argument('--config', type=open, action=LoadFromJSONFile, \
                        help='A JSON file storing experiment configuration. This can be found in the result directory')
    parser.add_argument('--datapath', type=str, \
                        help='Directory containing trained_models. Each pkl file contains matrices W2ext W2oo W2fut.')
    
    # Environment Options
    parser.add_argument('--env', type=str, required=False,  default='CartPole-v0', help='Gym environment')    
    add_boolean_option(parser, 'addrwd', default=False, help='Add rewards to observations')
    add_boolean_option(parser, 'addobs', default=False, help='Add observations to predictive states')
    parser.add_argument('--p_obs_fail', type=float, help='sensor failure probability for obs ')
    parser.add_argument('--T_obs_fail', type=int, help='max sensor failure time window for obs ')    
    parser.add_argument('--obsnoise', type=float, default=0.0, help='standard deviation for noise in obs space')
    #parser.add_argument('--actnoise', type=float, default=0.0, help='standard deviation for noise in act space')
    parser.add_argument('--obs_latency', type=int, default=0, help='observation latency in steps')
    parser.add_argument('--act_latency', type=int, default=0, help='action latency in steps ')
    add_boolean_option(parser, 'fullobs', default=False, help='Use fully observable environment state')
    add_boolean_option(parser, 'normalize_act', default=True, help='Scale actions within bounds [True]')
    add_boolean_option(parser, 'normalize_obs', default=False, help='Scale obs mean avg')
    add_boolean_option(parser, 'normalize_rwd', default=False, help='Scale rwds mean avg')
    add_boolean_option(parser, 'use_rllab', default=False, help='Use rllab environment')
    parser.add_argument('--critic', type=str, required=False, help='Replace the reward by a value function model')
    
    # Model Options
    parser.add_argument('--method', type=str, required=False, help='function to call.')
    parser.add_argument('--nh', nargs='+', type=int, default=[16],  help='number of hidden units. --nh L1 L2 ... number of hidden units for each layer')
    parser.add_argument('--nL', type=int, default=1, help='number of layers. 0- for linear')    
    parser.add_argument('--nn_act', type=str, default='relu', help='Activation function for feed-forward netwroks (relu/tanh)')
    parser.add_argument('--fut', type=int, help='future window')
    parser.add_argument('--past', type=int, help='past window')
    parser.add_argument('--reg', type=float, default=0.01, help='uniform regularization constant')
    parser.add_argument('--reg_s1a', type=float,   help='s1a regularization constant')
    parser.add_argument('--reg_s1b', type=float,   help='s1b regularization constant')
    parser.add_argument('--reg_s1c', type=float,   help='s1c regularization constant')
    parser.add_argument('--reg_s1div', type=float,   help='s1_divide regularization constant')
    parser.add_argument('--reg_ex', type=float,   help='Wext regularization constant')
    parser.add_argument('--reg_oo', type=float,   help='Woo regularization constant')
    parser.add_argument('--reg_pred', type=float,   help='pred regularization constant')
    parser.add_argument('--reg_filter', type=float,   help='filter regularization constant')   
    parser.add_argument('--dim', type=int,  help='PSR dimension (latente space dim).')
    parser.add_argument('--Hdim', type=int, default=1000, help='high Features dimension.')     
    parser.add_argument('--fext', type=str, help='feature extractor to call.')
    parser.add_argument('--kw', type=int, default=50, help='kernel width percentile [default:50]')
    parser.add_argument('--min_std', type=float, default=0.0, help='minimum policy std.')
    parser.add_argument('--gclip', type=float, default=10.0, help='gradient clipping')
    parser.add_argument('--r_max', type=float, default=10000.0, help='probability ratio limit max for TRPO and AltOp')
    
    add_boolean_option(parser, 'random_start', default=False, help='Start the psr with Random parameters [False]')
    parser.add_argument('--psr_iter', type=int, default=5, help='Number of rffpsr/rffpsr_rnn conjugate grad iterations')    
    parser.add_argument('--psr_state_norm', type=str, default='I', help="'I' identity, 'l2' l2 normalization, 'of' simple feature]")    
    parser.add_argument('--psr_smooth', type=str, default='I', help="'I' no op, 'interp_0.9' convex interpolation, 'search' do a search direction]")    
    parser.add_argument('--psr_cond', type=str, default='kbr', help="rff psr state update ['kbr':kernel bayes rule, 'kbrcg' KBR with conjugate gradient, 'kbrMIA' matrix inverse approximation vi Neumann Series, 'I' ignore Coo v= t_obs_feat]")    
  
    # PSR Refinement Options [Not used for now]
    parser.add_argument('--refine', type=int, default=0, help='number of model refinment iterations')    
    parser.add_argument('--valratio', type=float, help='training ratio. rest is for validation')
    #TODO: Fix description
    parser.add_argument('--roptimizer', type=str, default='adam', help='Optmizer for PSR refinement.') 
    parser.add_argument('--rstep', type=float, default=0.1, help='gradient descent step size')   
    #TODO: Fix description
    parser.add_argument('--valbatch', type=int, default=0, help='Refinement batch size')    
    parser.add_argument('--minrstep', type=float,default=1e-5, help='minimum refinement step')    
    
    # Data Collection Options
    parser.add_argument('--numtrajs', default=0, type=int, required=False,  help='number of trajectories per iteration')
    parser.add_argument('--numsamples', default=0, type=int, required=False,  help='number of samples per iteration')
    parser.add_argument('--mintrajlen', default=0, type=int, required=False,  help='number of samples per iteration')
    parser.add_argument('--rtmax', type=int, help='max number of retraining trajectories')   
    parser.add_argument('--len', type=int, required=False, help='Maximum of nominal trajectory length')    
    parser.add_argument('--leni', type=int, help='Maximum of initial trajectory length')    
    parser.add_argument('--initN', default=0, type=int, help='number of initial trajectories used to initialize model/policy')    
    parser.add_argument('--initS', default=0, type=int, help='number of initial samples used to initialize model/policy')    
    
    # Training Options
    parser.add_argument('--iter', type=int, required=False, default=500, help='number of training iterations')    
    parser.add_argument('--lr', type=float, required=False, default=1e-2, help='Learning rate for VR Reinforce')        
    parser.add_argument('--grad_step', type=float, default=1e-2, help='Learning rate for VR Reinforce')        
    parser.add_argument('--trpo_step', type=float, help='TRPO Step size')    
    parser.add_argument('--cg_opt', type=str, default='adam', help='gradient optimizer. Options: adam, adadelta,adagrad,RMSProp,sgd (default:adam)')        
    parser.add_argument('--gen', type=str,  help='train set generator function. Options: exp, reward, boot, or else: TrainSetGenerator')
    parser.add_argument('--wpred', type=float, default=1.0, help='weight on predictive error policy')
    parser.add_argument('--wdecay', type=float, default=1.0, help='weight on next error policy')
    parser.add_argument('--wpca', type=float, default=0.0, help='weight on pca projection for psr')
    parser.add_argument('--wrff', type=float, default=0.0, help='weight on rff projection feature matrices')
    parser.add_argument('--wrwd', type=float, default=1.0, help='weight on rewards for joint objective')
    parser.add_argument('--wkl', type=float, default = 1.0, help='weight on KL psr 1 step difference ')
    parser.add_argument('--wrwd_only', type=float, default=1.0, help='weight on rewards for individual objective')
    parser.add_argument('--ntr', type=int, help='retrain every ntrain iterations')        
    parser.add_argument('--repeat', type=int, default=1, help='number of times run each model to get stats results')    
    add_boolean_option(parser, 'saved_policy', default=False, help=' use a good initial policy [False]')
    add_boolean_option(parser, 'h0', default=True, help=' Optimize intial predictive state [False]')
    #parser.add_argument('--Uopt', type=bool, default=False, help=' Optimize projections [False]')    
    parser.add_argument('--init_policy', type=str,default='None', help=' use a good initial policy [default:None; OU,gauss]')            
    parser.add_argument('--pi_exp', type=str, default='None', help='exploration strategy [OU, gauss, None]')            
    parser.add_argument('--seed', type=int, default=0, help='experiment seed [0: default]')    
    parser.add_argument('--b', type=str, default='psr', help='type of baseline for PSR network [psr:default, obs,AR, none]')   
    parser.add_argument('--mse', type=float, help='retrain if mse incresed mse fold.')   
    parser.add_argument('--tmse', type=float, default=np.inf, help='retrain if mse > absolute threshold [default: not retrain (inf)].')   
    parser.add_argument('--freeze', type=int, default=1, help='freeze psr projections if retrain.')   
    parser.add_argument('--pi_update', type=str, default='proj', help='update policy parameters with proj:UnTU, reg:states regression  [default:proj]')
    parser.add_argument('--vr', type=str, default='VRrnn', help='Reinforce method  [default:VRrnn], VRpg, TRPO')
    #parser.add_argument('--opt_pca', type=bool, default=False, help='Include PCA projections in optimization')
    #parser.add_argument('--opt_rff', type=bool, default=False, help='Include RFF matrices in optimization')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for computing value function (cost-to-go)')
    parser.add_argument('--discount', type=float, default=1.0, help='discount factor for adam over iterations')
    add_boolean_option(parser, 'norm_g', default=True, help='Normalize policy gradients. [TRUE]')    
    parser.add_argument('--filter_w', type=int, default=1, help='Filtering model to run environment FMi[default:1]')
    parser.add_argument('--beta_lr', type=float, default=0.0, help='Update learning rate via variance normalization. exp.averaging ratio [0.0]')    
    parser.add_argument('--beta', type=float, default=0.1, help='Exp.averaging ratio for variance relative weights [0.1]')    
    parser.add_argument('--load_reactive', type=str, default='', help='Path to pickle file load reactive policy parameters')
    parser.add_argument('--decay1', type=float, default=0.0, help='decrease rate for loss1 importance')
    parser.add_argument('--decay2', type=float, default=0.0, help='decrease rate for loss2 importance')
    parser.add_argument('--threshold_var', type=float, default=0.0, help='threshold percentile for variance normalization')
    parser.add_argument('--var_clip', type=float, default=0.0, help='clip variance normalization [1/val, val]')
    add_boolean_option(parser, 'fix_psr', default=False, help='Fix PSR parameters. [FALSE]')    
    parser.add_argument('--hvec', type=str, default='exact', help='Hessian-vector multiplication method. (exact,fd)')
    
    # Output Options
    parser.add_argument('--tfile', type=str, default='results/', help='Directory containing test data.')
    add_boolean_option(parser, 'autosubdir', True, help='Create a subdir in result directory whose path is determined by experiment parameters.')
    parser.add_argument('--monitor', type=str, help='monitor file.')
    parser.add_argument('--vrate', type=int, default=1, help='Number of iterations after which a video is saved when monitoring is enabled')
    parser.add_argument('--irate', type=int, default=100000, help='Number of iterations after which a trajectory image is saved')
    parser.add_argument('--prate', type=int, default=50, help='Number of iterations after which results are pickled')
    add_boolean_option(parser, 'log', default=False, help='Report mse after updates and log UTU print slower')
    parser.add_argument('--logfile', type=str, help='Path to pickle file to save the output of the Logger')
    parser.add_argument('--loadfile', type=str,default='', help='Path to pickle file to save the output of the Logger')
    add_boolean_option(parser, 'render', default=False, help='Report mse after updates and log UTU print slower')
    
    # Debug Options
    add_boolean_option(parser, 'abort_err', default=False, help='Terminate upon exception. The default behavior is to ignore the current run')
    parser.add_argument('--dbg_len', nargs=2, type=int, help='Specifies a range of traj lengths to be chosen at random')
    add_boolean_option(parser, 'dbg_nobatchpsr', false_name='dbg_batchpsr', default=True, help='Do not use batched PSR updates')    
    add_boolean_option(parser, 'dbg_collapse', default=False, help='do not update if trajectory prediction collapsed to 0.')    
    parser.add_argument('--dbg_prederror', type=float, default=0.0, help='do not update if prediction error is larger than this value. Default[0.0] allways update')
    add_boolean_option(parser, 'dbg_mintrajrun', default=False, help='do not update if trajectory prediction collapsed to 0.')    
    add_boolean_option(parser, 'mask_state', default=False, help='set predictive states to 0.[Default:False]')    
    parser.add_argument('--dbg_reward', default=0.0, type=float, help='Control cost for Envwrapper for reward shapping. [Default:1.0 same as default openAI] ')    
    parser.add_argument('--powerg', default=2., type=float, help='Control cost for Envwrsquash gradietn to -1,1.[Default:False]')    
    add_boolean_option(parser, 'squashg', default=False, help='squash tanh gradient [Default:False]')    
    
    # Start with exploration trajectories 
    parser.add_argument('--exp_trajs', type=str, help='pickle file with exploration trajectories')
    return parser
       
if __name__ == '__main__': 
    parser = get_parser()       
    args = parser.parse_args()
    test_file = args.tfile + '/'
    
    if args.autosubdir:
        exclude_from_name = ['load','tfile','loadfile','iter','repeat','statenoise','monitor','vrate', 'seed','adam','irate', 'log','refine'\
                             'normalize_obs','normalize_rwd','normalize_act', 'random_start', 'Hdim','roptimizer', 'tmse','len',\
                             'abort_err','dbg_len','len', 'irate', 'refine','obsnoise','random_start','addrwd','addobs']
        subdir = ''.join([k[-4:]+str(v)+'_' for k,v in vars(args).iteritems() if v is not None and k not in exclude_from_name])
        test_file = test_file + subdir[:200] + '/'
            
    args.autosubdir = True # Restore default value to avoid unexpected behavior when running from config.
    mkpath(test_file)    
    setattr(args, 'file', test_file)
    #setattr(args, 'logfile', test_file+'log.log')
    json.dump(args.__dict__, open(test_file+'params','w'))
    
    args.rng = np.random.RandomState(args.seed)
    globalconfig.vars.args = args
    psr_lite.globalconfig.vars.args = args
   
    
#     functions={ 'gym_pred': test_rffpsr_planning.test_continuous_prediction,
#                 'gym_sim': test_rffpsr_planning.test_simulated_prediction,
#                 'gym_model': test_gym.test_continuous_model,
#                 'wave': test_rffpsr_planning.test_DS,
#                 'ode': test_rffpsr_planning.test_DS,
#                 'circulant': test_rffpsr_planning.test_DS,
#                 'lite-cont': test_policy_continuous,
#                 'lite-obs': test_policy_continuous,
#                 'obsVR': test_policy_continuous,
#                 'arVR':  test_policy_continuous,
#                 'obsVRdiscrete': test_policy_discrete,
#                }
#     import logging
#     logging.basicConfig(level=logging.DEBUG)
#     logger = logging.getLogger(__name__)
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     fh = logging.FileHandler(test_file+'log.log')
#     fh.setLevel(logging.DEBUG)
#     fh.setFormatter(formatter)
#     logger.addHandler(fh)
#     setattr(args, 'logger', logger)
#     logger.debug('Dumping results to:\n%s',test_file)
    if args.logfile is not None:
        Logger.instance().set_file(args.logfile)
    if args.repeat==1:
        args.trial = args.seed
        run_model(args, test_file, args.trial, loadfile=args.loadfile) 
    else:
        run_Nmodel(args, test_file, N=args.repeat, loadfile=args.loadfile)    
    Logger.instance().stop()
    
    
    
    
    ##examples:
    # python -m psr_models.tests.call_test --method ode --render 1 --numtrajs 500 --batch 100 --len 50
    # python -m psr_models.tests.call_test --method gym_pred --env Swimmer-v1 --render 1 --numtrajs 200 --batch 100 --len 50 --reg 1e-3 --rstep 1e-3 --monitor simple_exp
    # python -m psr_models.tests.call_test --method gym_pred --env CartPole-v0 --render 1 --numtrajs 1000 --batch 500 --len 200 --reg 1e-3 --rstep 1e-3
    # python -m psr_models.tests.call_test --method gym_model --env Swimmer-v1 --render 100 --numtrajs 300 --batch 100 --len 70 --reg 1e-3 --rstep 1e-4 --monitor 'short_seqs' --iter 100 --gen reward --nL 0 --wpred 1.0
    # python -m psr_models.tests.call_test --method gym_model --env CartPole-v0 --render 100 --numtrajs 1000 --batch 500 --len 200 --reg 1e-3 --rstep 5e-4 --iter 100 --gen reward --nh 2 --nL 20 20 --wpred 1.0
    # python call_test.test_policy_network --method lite-cont --env CartPole-v0 --numtrajs 20 --len 500 --past 10 --fut 10 --dim 50 --blindN 40 --Hdim 1000 --nL 1 --nh 10 --lr 1e-4 --wpred 10.0 --wgrad 0.1 --wdecay 0.1 --iter 100
