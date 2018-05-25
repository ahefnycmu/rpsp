#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 12:38:44 2017

@author: ahefny
"""
import numpy as np
import NN_policies
from envs.environments import *
from envs.simulators import *
import psr_lite.psr_lite.rnn_filter
import psr_lite.psr_lite.rffpsr
import psr_lite.psr_lite.feat_extractor
import psr_lite.psr_lite.rffpsr_rnn
import psr_lite.psr_lite.gru_filter
import psr_lite.psrlite_policy
from models import ObservableModel, FiniteHistoryModel, FiniteDeltaHistoryModel, ConstantModel, ZeroModel
import models
import policies
import psr_lite.psr_lite_vrpg
from policy_learn import learn_policy
from IPython import  embed
from test_utils.plot import call_plot, load_model, save_model, load_params, save_params
from psr_lite.psr_lite.noisy_model import NoisyModel
from policies import LinearPolicy
from envs.load_environments import load_environment
from psr_lite.psr_lite.rnn_model import RNN_trainer, BaseTrainer
import nn_policy_updaters
from nn_policy_updaters import *
from psr_policy_updaters import *
from collections import defaultdict
from time import time
import traceback            
from test_utils.plot import plot_trials 
from NN_policies import ContinuousExplorationPolicy,ContinuousPolicy
policy_saved_dir = 'psr_lite/datagen/policies/'
from explore.gaussian_strategy import GaussianStrategy
from explore.ou_strategy import OUStrategy

class structtype():
    pass

policy_updator={'VRrnn': defaultdict(lambda: NN_policies.VR_Reinforce_PolicyUpdater, 
                        {'lite-cont':psr_lite.psr_lite_vrpg.VR_Reinforce_RNN_PolicyUpdater,
                         'gru':psr_lite.psr_lite_vrpg.VR_Reinforce_RNN_PolicyUpdater,
                         'lstmVR': NN_policies.VR_Reinforce_RNN_PolicyUpdater }),
                'VRpg': defaultdict(lambda:VRPGPolicyUpdater, 
                        {'lite-cont':PSR_VRPGPolicyUpdater,
                         'gru':PSR_VRPGPolicyUpdater}),
                'TRPO': defaultdict(lambda:TRPOPolicyUpdater, 
                        {'lite-cont': PSR_TRPOPolicyUpdater,
                         'gru':PSR_VRPGPolicyUpdater}),
                'AltOp': defaultdict(lambda: PSR_AltOpt_TRPOPolicyUpdater, {}),
                'jointAltOp': defaultdict(lambda: PSR_JointAltOp_PolicyUpdater, {}) ,
                'jointVROp': defaultdict(lambda: PSR_JointVRPG_PolicyUpdater, {}),
                #'normVRpg': defaultdict(lambda: NormVRPG_PolicyUpdater,{}),
                #'jointTRPO': defaultdict(lambda: JointTRPO, {})
                }
get_policy=defaultdict(lambda:lambda *args,**kwargs: ContinuousPolicy(*args,**kwargs),\
                            {'OU': lambda *args,**kwargs: ContinuousExplorationPolicy(OUStrategy, *args,**kwargs), \
                            'gauss':lambda *args,**kwargs: ContinuousExplorationPolicy(GaussianStrategy, *args, **kwargs)
                            })

class Log(object):
    def __init__(self, args, filename, n=3, pred_model=None):
        self._pp = call_plot(name=filename,n=n, trial=args.trial)
        self._pred_model = pred_model
        self._args = args
        self._irate = args.irate
        self._last_err = np.inf
        self.retrain = False #retrain if mse
        if self._pred_model is not None: 
            self._U_old = self._pred_model.get_projs()
        self.avg_traj=[]
        self._results = {'act':[],'rewards':[],'rwd':[],'obs':[],'mse':[],'exp':filename,'rng':[], 'env_states':[]}
        
    def log_U(self, iter):
        from psr_lite.psr_lite.utils.log import Logger
        U_new = self._pred_model.get_projs()
        for (k,v) in U_new.items():
            d_min = min([U_new[k].shape[0], self._U_old[k].shape[0]])        
            Logger.instance().append('U_%s_%d' % (k, iter), np.dot(v[:d_min,:].T, self._U_old[k][:d_min,:]), print_out=self._args.log>1)
        self._U_old = U_new
        return
        
    def logger(self, i, trajs, res, track_delta=False):                
        # Output stats
        C = [np.sum(t.rewards) for t in trajs]
        m = np.mean(C); 
        s = np.std(C);  
        name='RT' if self.retrain else ''
        wdecay = 1.0 if self._args.wdecay is None else self._args.wdecay
        wpred = self._args.grad_step if self._args.wpred is None else self._args.wpred
        rwd_coeff = self._args.wrwd if self._args.wrwd>0.0 else self._args.wrwd_only
        wrwd = self._args.trpo_step if rwd_coeff is None else rwd_coeff
        res['best_vel_avg'] = np.mean(trajs[trajs[-1].bib].vel)
        res['best_vel_min'] = np.min(trajs[trajs[-1].bib].vel)
        res['best_vel_max'] = np.max(trajs[trajs[-1].bib].vel)
        res['best_rwd'] = np.sum(trajs[trajs[-1].bib].rewards)
        if self._pred_model is None:
            if (i%self._irate==0):
                #self._pp.plot_single(m,s)
                self._pp.plot_traj(trajs[0], trajs[0].obs)
                self._pp.plot(np.mean(res.get('cost1_avg',0.)),np.std(res.get('cost1_avg',0.)),
                              np.mean(res.get('fvel_avg',0.0)),np.std(res.get('fvel_avg',0.0)),
                              m,s, self.retrain,
                              label_2='vel')
        else:
            normalizer = float(wpred*wdecay**i) if wpred>0.0 else wdecay**i
            emse = (res.get('total_cost',0.)-wrwd*res.get('reinf_cost_avg',0.))/normalizer
            R = [emse]
            self._results['mse'].append(R)
            if track_delta:
                ##track difference between avg trajectory for exploration evaluation
                avg = np.zeros((self._args.numtrajs,self._args.len,trajs[0].obs.shape[1]))
                for k,t in enumerate(trajs):
                    avg[k,:t.obs.shape[0],:] = t.obs/float(t.obs.shape[0])
                self.avg_traj.append(np.sum(avg,axis=0))
                print '\t\tdelta_batch_avg:{} delta_prev_avg:{}'.format( np.linalg.norm(np.mean([(t.obs - self.avg_traj[-1][:t.obs.shape[0]])**2],axis=0)),
                                                                     np.linalg.norm(np.mean([(t.obs - 0.0 if len(self.avg_traj)<2 else self.avg_traj[-2][:t.obs.shape[0]])**2],axis=0)),
                                                                    )
            self._last_err = np.mean(np.copy(R))
            if (i%self._irate==0):
                try:
                    reinf = res.get('trpo_cost',0.)
                except KeyError:
                    reinf = res.get('cost2_avg',0.)
                self._pp.plot(np.mean(res.get('cost1_avg',0.)),np.std(res.get('cost1_avg',0.)),
                              np.mean(res.get('fvel_avg',0.0)),np.std(res.get('fvel_avg',0.0)),
                              m,s, self.retrain,
                              label_2='vel')
                tpred = self._pred_model.traj_predict_1s(trajs[0].states, trajs[0].act)
                self._pp.plot_traj(trajs[0], tpred,name=name)
        print 'reg:{} psr_step:{} rwd_w:{} past:{} fut:{}'.format(self._args.reg, self._args.grad_step, self._args.wrwd, self._args.past, self._args.fut)
        print '\t\t\t\t\t\t'+'\t\t\t\t\t\t'.join(['{}={}\n'.format(k,res.get(k, 0.0)) for k in res.keys()])
        self._results['rewards'].append([np.sum(t.rewards) for t in trajs])
        
        if (i%50==0):
            self._results['env_states'].append([trajs[trajs[-1].bib].env_states])
            self._results['rwd'].append([trajs[trajs[-1].bib].rewards])
            self._results['act'].append([trajs[trajs[-1].bib].act])
            self._results['rng'].append([trajs[trajs[-1].bib].rng])
            self._results['obs'].append([trajs[trajs[-1].bib].obs])
        if (i%(self._args.prate)==0):
            #save pickle results
            save_model(self._args.method+'_trial%d'%self._args.trial, self._args.flname, self._results, self._args)   
        return False #self.retrain

          
def get_train_generator(args):
    if args.gen is None:
        args.gen = 'train'
    trainset={'train':models.TrainSetGenerator(max_size=args.numtrajs if args.rtmax is None else args.rtmax),
              'boot':models.BootstrapTrainSetGenerator(batch=args.numtrajs, max_size=np.inf if args.rtmax is None else args.rtmax), 
              }
    return trainset[args.gen]
                        
def psr_model(args, data=[], **kwargs):
    X_obs, X_act = data
    
    if args.method == 'lite-cont':
        feats = psr_lite.psr_lite.feat_extractor.create_RFFPCA_featureset(args.Hdim,args.dim, pw=args.kw, rng=args.rng)

        if args.reg is None:
            lambda_psr = {'s1a':args.reg_s1a, 's1b':args.reg_s1b, 's1c':args.reg_s1c, 's1div':args.reg_s1div,
                's2ex':args.reg_ex, 's2oo':args.reg_oo, 'filter':args.reg_filter, 'pred':args.reg_pred}
        else:
            lambda_psr = psr_lite.psr_lite.rffpsr.uniform_lambda(args.reg)
        psr = psr_lite.psr_lite.rffpsr.RFFPSR(args.fut, args.past, args.dim, feature_set=feats,l2_lambda=lambda_psr,\
                                               psr_iter=args.psr_iter, psr_cond=args.psr_cond, psr_norm=args.psr_state_norm,\
                                               rng=args.rng)
                
        if args.random_start:
            psr.initialize_random(X_obs, X_act)
        elif args.loadfile<>'':
            pass #will load using psrnet
        else:
            train_data = psr.train(X_obs, X_act)
        psr.freeze()
        
        psrrnn = psr_lite.psr_lite.rffpsr_rnn.RFFPSR_RNN(psr, optimizer=args.roptimizer, optimizer_step=args.rstep ,optimizer_iterations=args.refine, \
                                                     optimizer_min_step=args.minrstep, rng=args.rng, opt_h0=args.h0, \
                                                     psr_iter=args.psr_iter, psr_cond=args.psr_cond, \
                                                     psr_norm=args.psr_state_norm, val_trajs=args.valbatch,\
                                                     opt_U=args.wpca, opt_V=args.wrff, psr_smooth=args.psr_smooth)
            
    elif args.method == 'gru':
        psr = None
        psrrnn = psr_lite.psr_lite.gru_filter.GRUFilter(args.dim, args.dim,  args.fut, \
                                                        optimizer=args.roptimizer, optimizer_step=args.rstep,
                                                        optimizer_iterations=args.refine, val_trajs=args.valbatch,\
                                                        optimizer_min_step=args.minrstep, rng=args.rng)
        
    else:
        assert False

    if args.addobs:
        print 'Extended model: obs_dim ', kwargs['x_dim']
        psrrnn = psr_lite.psr_lite.rnn_filter.ObsExtendedRNN(psrrnn, kwargs['x_dim'], args.filter_w, args.mask_state)
        
        
    if args.loadfile<>'':
        psrrnn._load(args.params['policy']['psrnet'])
    
    psrrnn.train(X_obs, X_act, on_unused_input='raise')
    psrtrainer = RNN_trainer( batch_gen=get_train_generator(args),\
                    log=1, pi_update=args.pi_update) 
    return psr, psrrnn, psrtrainer

def rff_cte_model(args, data=[], **kwargs):
    X_obs, X_act = data
    feats = psr_lite.psr_lite.feat_extractor.create_RFFPCA_featureset(args.Hdim,args.dim, pw=args.kw, rng=args.rng)

    if args.reg is None:
        lambda_psr = {'s1a':args.reg_s1a, 's1b':args.reg_s1b, 's1c':args.reg_s1c, 's1div':args.reg_s1div,
            's2ex':args.reg_ex, 's2oo':args.reg_oo, 'filter':args.reg_filter, 'pred':args.reg_pred}
    else:
        lambda_psr = psr_lite.psr_lite.rffpsr.uniform_lambda(args.reg)
    psr = psr_lite.psr_lite.rffpsr.RFFPSR(args.fut, args.past, args.dim, feature_set=feats,l2_lambda=lambda_psr,\
                                           cond_iter=args.icg, rng=args.rng)
    
    if args.random_start:
        psr.initialize_random(X_obs, X_act)
    else:
        train_data = psr.train(X_obs, X_act)
    psr.freeze()
    psrrnn = psr_lite.psr_lite.rffpsr_rnn.RFFPSR_RNN_dbug(psr, optimizer_iterations=args.refine, \
                                                     cond_iter=args.icg, rng=args.rng, opt_h0=args.h0, \
                                                     opt_U=args.wpca, opt_V=args.wrff)


    psrrnn.train(X_obs, X_act, on_unused_input='ignore')
    psrtrainer = BaseTrainer( batch_gen=get_train_generator(args)) 
    return psr, psrrnn, psrtrainer

def rff_obs_model(args,data=[], **kwargs):
    X_obs, X_act = data
    x_dim = X_obs[0].shape[1]
    feat_set = psr_lite.psr_lite.feat_extractor.create_RFFPCA_featureset(args.Hdim,args.dim, pw=args.kw, rng=args.rng)
    model = ObservableModel(obs_dim = args.dim); 

    rnnmodel = psr_lite.psr_lite.rnn_filter.RFFobs_RNN(model, fset=feat_set, opt_U=args.wpca, opt_V=args.wrff, dim=args.dim)
    rnnmodel.extract_feats(X_obs, X_act)
    rnnmodel.train(X_obs, X_act, on_unused_input='ignore')
    obstrainer = RNN_trainer(batch_gen=get_train_generator(args),\
                            log=1)
    return model, rnnmodel, obstrainer


def obs_model(args, x_dim=0, data=[], **kwargs):
    X_obs, X_act = data
    x_dim = X_obs[0].shape[1]
    model = ObservableModel(obs_dim = x_dim); 
    rnnmodel = psr_lite.psr_lite.rnn_filter.ObservableRNNFilter(model)
    rnnmodel.train(X_obs, X_act, on_unused_input='ignore')
    obstrainer = BaseTrainer(batch_gen=get_train_generator(args))
    return model, rnnmodel, obstrainer

def model_call(args, **kwargs):
    func={'lite-cont': psr_model, 'gru': psr_model, 'lite-obs':obs_model,
          'lite-rffobs':rff_obs_model, 'lite-cte':rff_cte_model}
    return func[args.method](args, **kwargs)
  
  
# def get_baseline(args):
#     if args.b=='None':
#         return False
#     if args.vr =='VRpg' or args.vr=='TRPO' or args.vr=='AltOp':
#         baseline = nn_policy_updaters.LinearBaseline()
#     else:
#         baseline = args.b
#     return baseline
       
       
def test_policy_continuous(args, flname, params=None):
    args.flname = flname      
    env, model_exp, pi_exp = load_environment(args)   #TODO: remove from envs not used

    #env.reset()
    (x_dim, a_dim) = env.dimensions
    
    output_dim = a_dim
    
    #if model_exp is None:
    model_exp = ObservableModel(x_dim)
    #if pi_exp is None:
    pi_exp = policies.RandomGaussianPolicy(x_dim, rng=args.rng)
    print 'dimension:', x_dim, a_dim
    
    baseline = args.b #get_baseline(args)    
    min_traj_length = getattr(args, 'mintrajlen', args.past+args.fut+2)
    #:wqmin_traj_length = max([args.past+args.fut+2, min_traj_length]) #ensure for PSRs correct length 
    fargs = [pi_exp]
    fkwargs={'baseline':baseline, 'lr':args.lr, 'beta_reinf':args.wrwd, 
            'beta_pred':args.wpred, 'beta_pred_decay':args.wdecay,
            'beta_only_reinf':args.wrwd_only, 'gamma':args.gamma,
            'grad_step': args.grad_step , 'trpo_step':args.trpo_step,
            'past': args.past, 'fut': args.fut, 'cg_opt':args.cg_opt,
            'max_traj_length':args.len, 'num_trajs': args.numtrajs,
            'normalize_grad':args.norm_g, 'hvec':args.hvec}            

    rnntrainer=None    
    print 'build updater ... ',args.method
    ''' run the observable model with reactive policy'''
    if args.method=='obsVR':
        model = model_exp                       
        pi_react = get_policy[args.pi_exp](x_dim = model.state_dimension, output_dim = output_dim, \
                                                num_layers = args.nL, nh = args.nh,
                                                activation=args.nn_act, rng=args.rng, min_std=args.min_std);
        fargs[0] = pi_react
        PiUpdator = policy_updator[args.vr][args.method](*fargs, **fkwargs);
        pp = Log(args, flname, n=3)      
    elif args.method=='cteVR':
        model = ConstantModel(x_dim)
        pi_react = NN_policies.ContinuousPolicy(x_dim = model.state_dimension, output_dim = output_dim,
                                                num_layers = args.nL, nh = args.nh, 
                                                activation=args.nn_act, rng=args.rng, min_std=args.min_std);
        fargs[0] = pi_react
        PiUpdator = policy_updator[args.vr][args.method](*fargs, **fkwargs);
        pp = Log(args, flname, n=1)   
    elif args.method=='arVR':
        model = FiniteHistoryModel(obs_dim=x_dim, past_window=args.past)
        #model = ZeroModel(obs_dim=x_dim)                                       
        pi_react = get_policy[args.pi_exp](x_dim = model.state_dimension, output_dim = output_dim, \
                                                num_layers = args.nL, nh = args.nh,
                                                activation=args.nn_act, rng=args.rng, min_std=args.min_std);
        fargs[0] = pi_react
        PiUpdator = policy_updator[args.vr][args.method](*fargs, **fkwargs);
        pp = Log(args, flname, n=3)
    elif args.method=='deltaArVR':
        model = FiniteDeltaHistoryModel(obs_dim=x_dim, past_window=args.past, dt=env.dt)
        pi_react = NN_policies.ContinuousPolicy(x_dim = model.state_dimension, output_dim = output_dim,
                                                num_layers = args.nL, nh = args.nh,
                                                activation=args.nn_act, rng=args.rng, min_std=args.min_std);
        fargs[0] = pi_react
        PiUpdator = policy_updator[args.vr][args.method](*fargs, **fkwargs);
        pp = Log(args, flname, n=1)
    elif args.method=='rnnVR':
        raise NotImplementedError     
    elif args.method=='lstmVR':
        model = model_exp
        pi_react = NN_policies.RNN_Continuous_Policy(x_dim=model.state_dimension, \
                    a_dim = None, output_dim=output_dim,nh = args.nh[0], LSTM = True, \
                    rng=args.rng, ext=True, min_std=args.min_std) #nh=64
        fargs[0] = pi_react
        PiUpdator = policy_updator[args.vr][args.method](*fargs, clips=[-5,5], **fkwargs)
        pp = Log(args, flname, n=1)
    elif args.method=='psr_contVR':
        raise NotImplementedError
    elif args.method[:4]=='lite' or args.method == 'gru': #TODO: Need better method naming        
        ''' run the psr network with obs model or psr model'''
        model = ObservableModel(obs_dim=x_dim)               

        X_obs, X_act = get_exploration_trajs(args, model_exp, env, output_dim, min_traj_length)    
        print 'TEST REPRODUCIBILITY', X_obs[0][0]
        tic=time()
    
        rffpsr_model, rnnmodel, rnntrainer = model_call(args, data=[X_obs,X_act], x_dim=x_dim)
        print('INIT RPSP without refinement takes:', time()-tic)
        #state_dim = rffpsr_model.state_dimension
        state_dim = rnnmodel.state_dimension
        print 'STATE DIMENSION',state_dim
        
        pi_react = get_policy[args.pi_exp](x_dim = state_dim, output_dim = output_dim, \
                                                num_layers = args.nL, nh = args.nh,
                                                activation=args.nn_act, rng=args.rng, min_std=args.min_std);

                                                
        if  isinstance(rnnmodel, psr_lite.psr_lite.rffpsr_rnn.RFFPSR_RNN):
            pi = psr_lite.psrlite_policy.RFFPSRNetworkPolicy(rnnmodel, pi_react, np.zeros((output_dim)))
        else:
            pi = psr_lite.psrlite_policy.PSRLitePolicy(rnnmodel, pi_react, np.zeros((output_dim)))
            
        fargs[0] = pi
        pp = Log(args, flname, pred_model=rnnmodel)
        print 'Building policy psr graph'

        tic = time()        
        PiUpdator = policy_updator[args.vr][args.method](*fargs,**fkwargs)
        print 'took ', time()-tic
    print 'done building updater'
    print 'len:',args.len, 'num trajs:', args.numtrajs, 'iter:',args.iter
    
    
    def run_experiment():
        if args.loadfile<>'':
            PiUpdator._load(args.params)
        elif args.load_reactive<>'':
            re_params = load_params(args.load_reactive)
            try:
                PiUpdator._policy._policy._load(re_params)
            except AttributeError:
                pass
       
        learn_policy(PiUpdator, model, env, min_traj_length=min_traj_length if args.dbg_mintrajrun else 0,
            max_traj_len = args.len, num_trajs = args.numtrajs, 
            num_samples=args.numsamples, num_iter = args.iter, 
            logger=pp.logger,
            trainer=rnntrainer,
            freeze=args.freeze)
       
       
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
    pp._results['params'] = PiUpdator._save()
    if args.addobs or args.method=='arVR':
        try:
            re_params = PiUpdator._policy._policy._save()
        except AttributeError:
            re_params = PiUpdator._policy._save()
        save_params(re_params, 're_pi_{}.pkl'.format(args.seed), args.tfile)
    env.close()
    return pp._results


def get_exploration_trajs(args, model_exp, env, output_dim, min_traj_length):
    
    if args.saved_policy :
        results, save_args = load_model( args.env, policy_saved_dir)
        pi_K = results[0]['K']
        pi = policies.LinearPolicy(pi_K, 0, rng=args.rng)
        exp_trajs = env.run(model_exp, pi, args.len, num_trajs=args.initN)
        col_trajs = [(t.obs, t.act) for t in exp_trajs]
        X_obs_good = [c[0] for c in col_trajs]
        X_act_good = [c[1] for c in col_trajs] 
    else:
        X_obs_good = []
        X_act_good = []
        
    re_params={}; pi_exp = policies.RandomGaussianPolicy(output_dim, rng=args.rng)
    #import pickle as p
    try:
        if args.exp_trajs is None: raise IOError
        print args.exp_trajs
        from rllab_scripts.rllab_explore import load_rllab_trajs
        exp_trajs = load_rllab_trajs(args.exp_trajs)
        print 'loaded exploration trajectories from: ',args.exp_trajs
    except IOError:
        if args.load_reactive<>'':
            re_params = load_params(args.load_reactive)
            if len(re_params)>0:
                x_dim = env.dimensions[0]
                model_exp = FiniteHistoryModel(obs_dim=x_dim, past_window=args.filter_w)
                state_dim = model_exp.state_dimension
                re_params['layer_id_0_W'] =  re_params['layer_id_0_W'][:,-state_dim:]
                pi_exp = get_policy[args.init_policy](x_dim = state_dim, output_dim = output_dim, \
                                                    num_layers = args.nL, nh = args.nh,
                                                    activation=args.nn_act, rng=args.rng, min_std=args.min_std)
                pi_exp._load(re_params)
        elif args.init_policy=='OU':
            pi_exp = policies.OUPolicy(output_dim,rng=args.rng)
            print 'using OU exploration policy'
        
            
        leni = args.len if args.leni is None else args.leni
        exp_trajs = env.run(model_exp, pi_exp, leni, render=False, 
                            min_traj_length=min_traj_length, num_trajs=args.initN,
                            num_samples=args.initS)
    print 'Using %d exp trajectories.'%len(exp_trajs)
    col_trajs = [(t.obs, t.act) for t in exp_trajs]
    X_obs_rnd = [c[0] for c in col_trajs]
    X_act_rnd = [c[1] for c in col_trajs]
    X_obs = X_obs_rnd + X_obs_good
    X_act = X_act_rnd + X_act_good
    return X_obs, X_act




