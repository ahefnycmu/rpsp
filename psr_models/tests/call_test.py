# -*- coding: utf-8 -*-
'''
Created on Tue Feb 7 12:16:30 2017

@author: zmarinho
'''

import argparse
import numpy as np
from distutils.dir_util import mkpath
import importlib
from IPython import embed
import test_rffpsr_planning
from psr_models.tests import test_gym
from psr_models.tests import test_rffpsr_planning


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
    parser.add_argument('--maxtrajs', type=int, default=2000, help='max number of trajectories per iterations')
    parser.add_argument('--batch', type=int, default=None, help='max number of training trajectories.')
    parser.add_argument('--dim', type=int, default=40, help='PSR dimension (latente space dim).')
    parser.add_argument('--Hdim', type=int, default=1000, help='high Features dimension.')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate for VR Reinforce')
    parser.add_argument('--len', type=int, default=100, help='Maximum of nominal trajectory length')
    parser.add_argument('--env', type=str, default='CartPole-v0', help='Gym environment')
    parser.add_argument('--rstep', type=float, default=5e-4, help='gradient descent step size')
    parser.add_argument('--tfile', type=str, default='results/', help='Directory containing test data.')
    parser.add_argument('--monitor', type=str, default='', help='monitor file.')
    parser.add_argument('--render', type=int, default=100, help='render final trajectories number.')
    parser.add_argument('--method', type=str, default='', help='function to call.')
    parser.add_argument('--fext', type=str, default='rff', help='feature extractor to call.')
    parser.add_argument('--gen', type=str, default='', help='train set generator function. Options: exp, reward, boot, or else: TrainSetGenerator')
    parser.add_argument('--nh', nargs='+', type=int, default=[16], help='number of hidden units. --nh L1 L2 ... number of hidden units for each layer')
    parser.add_argument('--nL', type=int, default=0, help='number of layers. 0- for linear')
    parser.add_argument('--wpred', type=float, default=1.0, help='weight on predictive error policy')
    parser.add_argument('--wnext', type=float, default=0.0, help='weight on next error policy')
    parser.add_argument('--blindN', type=int, default=100, help='initial blind policy start samples')
    parser.add_argument('--kw', type=int, default=70, help='kernel width')
    parser.add_argument('--ntrain', type=int, default=1000, help='fully train in the first iterations')
    parser.add_argument('--valratio', type=float, default=0.7, help='training ratio. rest is for validation')
    parser.add_argument('--optimizer', type=str, default='sgd', help='feature extractor to call.')
    parser.add_argument('--valbatch', type=int, default=5, help='fully train in the first iterations')
    parser.add_argument('--adam', type=bool, default=True, help='Refine Adam is RefineModelGD numeric.')
    parser.add_argument('--minrstep', type=float, default=1e-6, help='minimum refinement step')
    
    args = parser.parse_args()
    
    test_file = args.tfile+'/%s_%s/d%d_f%d_p%d_rff%d_r%.2e_iter%d_R%d_N%d_B%d_M%d_lr%.1e_l%d_rstp%.1e_%s_nh%d_nL%s_w%.2f_fext%s_kw%d_train%d_next%.2f_%d/'%(\
                args.env, args.method, args.dim, args.fut, args.past, args.Hdim, args.reg, args.iter, args.refine,\
                args.numtrajs, args.batch, args.maxtrajs, args.lr, args.len, args.rstep, args.gen, args.nL, str(args.nh),\
                args.wpred, args.fext, args.kw, args.ntrain, args.wnext, args.blindN)

    print 'Dumping results to:\n%s'%test_file
    mkpath(test_file)
    #test_continuous_model(args, test_file)
    
    functions={ 'gym_pred': test_rffpsr_planning.test_continuous_prediction,
                'gym_sim': test_rffpsr_planning.test_simulated_prediction,
                'gym_model': test_gym.test_continuous_model,
                'wave': test_rffpsr_planning.test_DS,
                'ode': test_rffpsr_planning.test_DS,
                'circulant': test_rffpsr_planning.test_DS
               }
    
    result = functions[args.method](args, test_file)
    
    
    ##examples:
    # python -m psr_models.tests.call_test --method ode --render 1 --numtrajs 500 --batch 100 --len 50
    # python -m psr_models.tests.call_test --method gym_pred --env Swimmer-v1 --render 1 --numtrajs 200 --batch 100 --len 50 --reg 1e-3 --rstep 1e-3 --monitor simple_exp
    # python -m psr_models.tests.call_test --method gym_pred --env CartPole-v0 --render 1 --numtrajs 1000 --batch 500 --len 200 --reg 1e-3 --rstep 1e-3
    # python -m psr_models.tests.call_test --method gym_model --env Swimmer-v1 --render 100 --numtrajs 300 --batch 100 --len 70 --reg 1e-3 --rstep 1e-4 --monitor 'short_seqs' --iter 100 --gen reward --nL 0 --wpred 1.0
    # python -m psr_models.tests.call_test --method gym_model --env CartPole-v0 --render 100 --numtrajs 1000 --batch 500 --len 200 --reg 1e-3 --rstep 5e-4 --iter 100 --gen reward --nh 2 --nL 20 20 --wpred 1.0