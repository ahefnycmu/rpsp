#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 12:38:44 2017

@author: ahefny, zmarinho
"""
import traceback
from collections import defaultdict
from time import time
import numpy as np

import models
import policies
import psr_lite.psr_lite.feat_extractor
import psr_lite.psr_lite.gru_filter
import psr_lite.psr_lite.rffpsr
import psr_lite.psr_lite.rffpsr_rnn
import psr_lite.psr_lite.rnn_filter
import psr_lite.psr_lite_vrpg
import psr_lite.psrlite_policy
from NN_policies import ContinuousExplorationPolicy, ContinuousPolicy
from envs.load_environments import load_environment
from explore.gaussian_strategy import GaussianStrategy
from models import ObservableModel, FiniteHistoryModel
from nn_policy_updaters import VRPGPolicyUpdater
from policy_learn import learn_policy
from psr_policy_updaters import PSR_VRPGPolicyUpdater, PSR_AltOpt_TRPOPolicyUpdater
from test_utils.log import Log
from test_utils.plot import load_model, load_params, save_params


class structtype():
    pass


policy_updater = {
    # 'VRrnn': defaultdict(lambda: NN_policies.VR_Reinforce_PolicyUpdater,
    #         {'lite-cont':psr_lite.psr_lite_vrpg.VR_Reinforce_RNN_PolicyUpdater,
    #          'gru':psr_lite.psr_lite_vrpg.VR_Reinforce_RNN_PolicyUpdater,
    #          'lstmVR': NN_policies.VR_Reinforce_RNN_PolicyUpdater }),
    'VRpg': defaultdict(lambda: VRPGPolicyUpdater,
                        {'lite-cont': PSR_VRPGPolicyUpdater,
                         'gru': PSR_VRPGPolicyUpdater}),
    'AltOp': defaultdict(lambda: PSR_AltOpt_TRPOPolicyUpdater, {}),
}

get_policy = defaultdict(lambda: lambda *args, **kwargs: ContinuousPolicy(*args, **kwargs),
                         {'gauss':lambda *args, **kwargs: ContinuousExplorationPolicy(GaussianStrategy, *args, **kwargs)
                          }
                         )


def get_train_generator(args):
    if args.gen is None:
        args.gen = 'train'
    trainset = {'train': models.TrainSetGenerator(max_size=args.numtrajs if args.rtmax is None else args.rtmax),
                'boot': models.BootstrapTrainSetGenerator(batch=args.numtrajs,
                                                          max_size=np.inf if args.rtmax is None else args.rtmax),
                }
    return trainset[args.gen]


def model_call(args, **kwargs):
    func = {'lite-cont': rpsp_filter, 'gru': rpsp_filter, 'lite-obs': obs_model}#, 'lite-rffobs': rff_obs_model}
    return func[args.method](args, **kwargs)


'''
Filter types
'''
def load_rpsp_filter(args, data=[]):
    X_obs, X_act = data
    feats = psr_lite.psr_lite.feat_extractor.create_RFFPCA_featureset(args.Hdim, args.dim, pw=args.kw, rng=args.rng)

    if args.reg is None:
        lambda_psr = {'s1a': args.reg_s1a, 's1b': args.reg_s1b, 's1c': args.reg_s1c, 's1div': args.reg_s1div,
                      's2ex': args.reg_ex, 's2oo': args.reg_oo, 'filter': args.reg_filter, 'pred': args.reg_pred}
    else:
        lambda_psr = psr_lite.psr_lite.rffpsr.uniform_lambda(args.reg)
    psr = psr_lite.psr_lite.rffpsr.RFFPSR(args.fut, args.past, args.dim, feature_set=feats, l2_lambda=lambda_psr,
                                          psr_iter=args.psr_iter, psr_cond=args.psr_cond,
                                          psr_norm=args.psr_state_norm,
                                          rng=args.rng)

    if args.random_start:
        psr.initialize_random(X_obs, X_act)
    elif args.loadfile <> '':
        pass  # will load using psrnet
    else:
        train_data = psr.train(X_obs, X_act)
    psr.freeze()
    psrrnn = psr_lite.psr_lite.rffpsr_rnn.RFFPSR_RNN(psr, optimizer=args.roptimizer, optimizer_step=args.rstep,
                                                     optimizer_iterations=args.refine,
                                                     optimizer_min_step=args.minrstep, rng=args.rng, opt_h0=args.h0,
                                                     psr_iter=args.psr_iter, psr_cond=args.psr_cond,
                                                     psr_norm=args.psr_state_norm, val_trajs=args.valbatch,
                                                     psr_smooth=args.psr_smooth)
    return psrrnn, psr, feats


def load_gru_filter(args, **kwargs):
    model = None
    rnngru = psr_lite.psr_lite.gru_filter.GRUFilter(args.dim, args.dim, args.fut,
                                                    optimizer=args.roptimizer, optimizer_step=args.rstep,
                                                    optimizer_iterations=args.refine, val_trajs=args.valbatch,
                                                    optimizer_min_step=args.minrstep, rng=args.rng)
    return rnngru, model, None


'''
Specify RPSP filters
'''
def rpsp_filter(args, data=[], **kwargs):
    if args.method == 'lite-cont':
        filter, model, feats = load_rpsp_filter(args, data=data)
    elif args.method == 'gru':
        filter, model, feats = load_gru_filter(args, data=data)

    if args.addobs:
        print 'Extended model: obs_dim ', kwargs['x_dim']
        filter = psr_lite.psr_lite.rnn_filter.ObsExtendedRNN(filter, kwargs['x_dim'], args.filter_w, args.mask_state)

    if args.loadfile != '':
        filter._load(args.params['policy']['psrnet'])
    filter.train(*data, on_unused_input='raise')
    return model, filter


# def rff_obs_model(args, data=[], **kwargs):
#     X_obs, X_act = data
#     x_dim = X_obs[0].shape[1]
#     feat_set = psr_lite.psr_lite.feat_extractor.create_RFFPCA_featureset(args.Hdim, args.dim, pw=args.kw, rng=args.rng)
#     model = ObservableModel(obs_dim=args.dim);
#
#     filter = psr_lite.psr_lite.rnn_filter.RFFobs_RNN(model, fset=feat_set, opt_U=args.wpca, opt_V=args.wrff,
#                                                        dim=args.dim)
#     filter.extract_feats(X_obs, X_act)
#     filter.train(X_obs, X_act, on_unused_input='ignore')
#     return model, filter


def obs_model(args, data=[], **kwargs):
    X_obs, X_act = data
    x_dim = X_obs[0].shape[1]
    model = ObservableModel(obs_dim=x_dim)
    filter = psr_lite.psr_lite.rnn_filter.ObservableRNNFilter(model)
    filter.train(X_obs, X_act, on_unused_input='ignore')
    return model, filter


'''
Policy networks
'''
def load_observable_policy(args, model_exp, **kwargs):
    model = model_exp
    pi_react = get_policy[args.pi_exp](x_dim=model.state_dimension, output_dim=args.a_dim,
                                       num_layers=args.nL, nh=args.nh,
                                       activation=args.nn_act, rng=args.rng, min_std=args.min_std)
    PiUpdater = policy_updater[args.vr][args.method](pi_react, **kwargs)
    pp = Log(args, args.flname, n=3)
    return model, PiUpdater, pp


def load_finite_mem_policy(args, model_exp, **kwargs):
    model = FiniteHistoryModel(obs_dim=args.x_dim, past_window=args.past)
    pi_react = get_policy[args.pi_exp](x_dim=model.state_dimension, output_dim=args.a_dim,
                                       num_layers=args.nL, nh=args.nh,
                                       activation=args.nn_act, rng=args.rng, min_std=args.min_std)
    PiUpdater = policy_updater[args.vr][args.method](pi_react, **kwargs)
    pp = Log(args, args.flname, n=3)
    return model, PiUpdater, pp


def load_rpsp_policy(args, model_exp, **kwargs):
    model = ObservableModel(obs_dim=args.x_dim)

    X_obs, X_act = get_exploration_trajs(args,
                                         model_exp,
                                         kwargs.get('env'),
                                         args.a_dim,
                                         kwargs.get('min_traj_length'))
    tic = time()
    psr, filter = model_call(args, data=[X_obs, X_act], x_dim=args.x_dim)
    print 'INIT RPSP without refinement takes:', time() - tic
    state_dim = filter.state_dimension
    pi_react = get_policy[args.pi_exp](x_dim=state_dim, output_dim=args.a_dim,
                                       num_layers=args.nL, nh=args.nh,
                                       activation=args.nn_act, rng=args.rng,
                                       min_std=args.min_std)
    if isinstance(filter, psr_lite.psr_lite.rffpsr_rnn.RFFPSR_RNN):
        pi = psr_lite.psrlite_policy.RFFPSRNetworkPolicy(filter, pi_react, np.zeros((args.a_dim)))
    else:
        pi = psr_lite.psrlite_policy.PSRLitePolicy(filter, pi_react, np.zeros((args.a_dim)))

    pp = Log(args, args.flname, pred_model=filter)
    print 'Building policy psr graph'

    tic = time()
    PiUpdater = policy_updater[args.vr][args.method](pi, **kwargs)
    print 'took ', time() - tic
    return model, PiUpdater, pp


'''
Test function
'''
def test_policy_continuous(args, flname, params=None):
    args.flname = flname
    env = load_environment(args)
    (x_dim, a_dim) = env.dimensions
    args.a_dim = a_dim
    args.x_dim = x_dim
    print 'dimension:', x_dim, a_dim
    model_exp = ObservableModel(x_dim)
    pi_exp = policies.RandomGaussianPolicy(x_dim, rng=args.rng)
    baseline = args.b
    min_traj_length = getattr(args, 'mintrajlen', args.past + args.fut + 2)
    PiUpdater = None
    fkwargs = {'baseline': baseline, 'lr': args.lr, 'beta_reinf': args.wrwd,
               'beta_pred': args.wpred, 'beta_pred_decay': args.wdecay,
               'beta_only_reinf': args.wrwd_only, 'gamma': args.gamma,
               'grad_step': args.grad_step, 'trpo_step': args.trpo_step,
               'past': args.past, 'fut': args.fut, 'cg_opt': args.cg_opt,
               'max_traj_length': args.len, 'num_trajs': args.numtrajs,
               'normalize_grad': args.norm_g, 'hvec': args.hvec,
               'env': env, 'min_traj_len':min_traj_length}
    print 'build updater ... ', args.method
    ''' run the observable model with reactive policy'''

    if args.method == 'obsVR':
        model, PiUpdater, pp = load_observable_policy(args, model_exp, **fkwargs)
    elif args.method == 'arVR':
        model, PiUpdater, pp = load_finite_mem_policy(args, model_exp, **fkwargs)
    else:
        ''' run the psr network with obs model or psr model'''
        model, PiUpdater, pp = load_rpsp_policy(args, model_exp, **fkwargs)
    print 'done building updater'
    print 'len:', args.len, 'num trajs:', args.numtrajs, 'iter:', args.iter

    def run_experiment():
        if args.loadfile <> '':
            PiUpdater._load(args.params)
        elif args.load_reactive <> '':
            re_params = load_params(args.load_reactive)
            try:
                PiUpdater._policy._policy._load(re_params)
            except AttributeError:
                pass

        learn_policy(PiUpdater, model, env, min_traj_length=min_traj_length if args.dbg_mintrajrun else 0,
                     max_traj_len=args.len, num_trajs=args.numtrajs,
                     num_samples=args.numsamples, num_iter=args.iter,
                     logger=pp.logger)

    if args.abort_err:
        run_experiment()
    else:
        try:
            run_experiment()
        except AssertionError as exc:
            print 'WARNING: Got AssertionError !'
            print 'Message: %s' % exc.message
            print 'Stacktrace:'
            traceback.print_exc()
            return None
    pp._results['params'] = PiUpdater._save()
    if args.addobs or args.method == 'arVR':
        try:
            re_params = PiUpdater._policy._policy._save()
        except AttributeError:
            re_params = PiUpdater._policy._save()
        save_params(re_params, 're_pi_{}.pkl'.format(args.seed), args.tfile)
    env.close()
    return pp._results


def get_exploration_trajs(args, model_exp, env, output_dim, min_traj_length):
    X_obs_good = []
    X_act_good = []
    pi_exp = policies.RandomGaussianPolicy(output_dim, rng=args.rng)
    if args.load_reactive <> '':
        re_params = load_params(args.load_reactive)
        if len(re_params) > 0:
            x_dim = env.dimensions[0]
            model_exp = FiniteHistoryModel(obs_dim=x_dim, past_window=args.filter_w)
            state_dim = model_exp.state_dimension
            re_params['layer_id_0_W'] = re_params['layer_id_0_W'][:, -state_dim:]
            pi_exp = get_policy[args.init_policy](x_dim=state_dim, output_dim=output_dim, num_layers=args.nL,
                                                  nh=args.nh, activation=args.nn_act, rng=args.rng, min_std=args.min_std)
            pi_exp._load(re_params)

    leni = args.len if args.leni is None else args.leni
    exp_trajs = env.run(model_exp, pi_exp, leni, render=False,
                        min_traj_length=min_traj_length, num_trajs=args.initN,
                        num_samples=args.initS)
    print 'Using %d exp trajectories.' % len(exp_trajs)
    col_trajs = [(t.obs, t.act) for t in exp_trajs]
    X_obs_rnd = [c[0] for c in col_trajs]
    X_act_rnd = [c[1] for c in col_trajs]
    X_obs = X_obs_rnd + X_obs_good
    X_act = X_act_rnd + X_act_good
    return X_obs, X_act
