#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function

"""
Created on Tue Apr 25 20:07:55 2017

@author: ahefny, zmarinho
"""

from collections import OrderedDict

import numpy as np
import numpy.linalg as npla
import theano
import theano.tensor as T

from rpsp import globalconfig
from rpsp.policy_opt.SGD_opt import optimizers
from rpsp.policy_opt.cg_optimizer import ConstrainedOptimizer, DefaultConstraintOptimizerOps
from rpsp.policy_opt.policy_learn import BasePolicyUpdater
from rpsp.rpspnets.psr_lite.psr_base import AutoRegressiveControlledModel
from rpsp.rpspnets.psr_lite.utils.nn import tf_get_normalized_grad_per_param

class Baseline:
    def get_value_fn(self, trajs, traj_info):
        '''
        Returns a matrix representing the value function evaluated at each step
        in given trajectories.
        '''
        return NotImplementedError

class ZeroBaseline(Baseline):
    def get_value_fn(self, trajs, traj_info):
        return np.zeros(traj_info['ctg'].shape)
        
class LinearBaseline(Baseline):
    def __init__(self, filtering_model = None, reg=1e-7):
        self._filtering_model = filtering_model
        self._reg = reg
        self._CXY = None
        self._CXX = None
        
    def _compute_states(self, trajs):
        if self._filtering_model is None:
            return [t.prestates for t in trajs]
        else:
            traj_obs = [t.obs for t in trajs]
            traj_act = [t.act for t in trajs]
            self._filtering_model.train(traj_obs,traj_act)
            return [self._filtering_model.compute_pre_states(t) for t in trajs]
        
    def get_value_fn(self, trajs, traj_info):
        # Upate prediction model
        states = self._compute_states(trajs)
        states = [np.hstack((s,np.ones((s.shape[0],1)))) for s in states] # Add intercept
        ctg = traj_info['ctg']
        length = traj_info['length']
        
        if self._CXY is None:
            sdim = states[0].shape[1]
            self._CXX = np.zeros((sdim,sdim))
            self._CXY = np.zeros(sdim)
          
        p = 0.1            
        self._CXX *= p
        self._CXY *= p    
                        
        for i in xrange(len(trajs)):            
            s = states[i]
            c = ctg[i,:length[i]]
            
            self._CXX += (1-p) * s.T.dot(s)
            self._CXY += (1-p) * s.T.dot(c)
                                    
        W = npla.solve(self._CXX + np.eye(self._CXX.shape[0]) * self._reg, self._CXY)
        
        out = np.zeros(ctg.shape)
        
        for i in xrange(len(trajs)):
            out[i][:length[i]] = states[i].dot(W)
            
        return out


default_baselines= {'obs':lambda past,fut: LinearBaseline(AutoRegressiveControlledModel(1,1)),
                    'AR':lambda past,fut: LinearBaseline(AutoRegressiveControlledModel(1,5)),
                    'psr':lambda past,fut: LinearBaseline(),
                    'None':lambda past,fut: ZeroBaseline(),
                    False:lambda past,fut: ZeroBaseline(),
                    None:lambda past,fut: ZeroBaseline()}

##############################################################################
'''
Utility functions for symbolic trajectory information (t_traj_info).
An object commonly used below is t_traj_info which is a dictionary of symbolic varibales
representing features of all trajectories in (NxTxd) tensors where N is number of trajectories,
T is trajectory length and d is feature dimension. Scalar features (e.g. rewards) are represented 
by NxT matrices. A special entry 'length' is an integer vector storing the actual length of each trajectory.
'''

def tf_call_traj_info_function(t_traj_fn, info_keys, args): 
    '''
    Given a list of arguments. Call a function that expects a t_traj_info
    dictionary.
    '''       
    single_traj_info = OrderedDict(zip(info_keys, args))
    return t_traj_fn(single_traj_info)

def tf_cat_traj_info(t_traj_info):
    '''
    Concatenates trajectory info of multiple trajectories in a single trajectory
    '''
    def reshape_tensor(x):        
        if x.ndim == 2:
            y = x.reshape((-1,))
        elif x.ndim == 3:
            y = x.reshape((-1,x.shape[-1]))
        else:
            return None
         
        if x.name is not None:    
            y.name = x.name + '_cat'
        return y
        
    reshaped_traj = OrderedDict((k,reshape_tensor(v)) for (k,v) in t_traj_info.items())    
    reshaped_traj['length'] = reshaped_traj['mask'].shape[0]
    return reshaped_traj

def _create_mean_function(t_traj_info, t_single_traj_fn, num_trajs = None, info_keys = None):
    '''        
    Given a symbolic function t_single_traj_fn(t_single_traj_info)
    constructs a symbolic function that computes the mean over a number of trajectories.
    
    The parameter 'info_keys' specifies the elements in traj_info that needs to be passed to the function.
    If not specified, all keys are passed.
    '''    
    if info_keys is None:
        info_keys = t_traj_info.keys()

    call_fn = lambda *args : tf_call_traj_info_function(t_single_traj_fn, info_keys, args)
        
    if num_trajs is None:
        # Use scan function
        ccs,_ = theano.scan(fn=call_fn,
                        sequences=t_traj_info.values(),                            
                        n_steps=t_traj_info.values()[0].shape[0])
        return T.mean(ccs, axis=0)                
    else:        
        # Use for loop for faster execution. 
        ccs = [None] * num_trajs
        for i in xrange(num_trajs):
            t = [x[i] for x in t_traj_info.values()]
            ccs[i] = call_fn(*t)
                
        return T.mean(T.stack(ccs), axis=0)        
        
def create_mean_function_nonseq(t_traj_info, t_single_traj_fn):
    '''        
    A faster version of _create_mean_function where t_single_traj_fn is non-sequential and therefore
    we can stack multiple trajectories as a single trajectory. 
    Note that t_single_traj_fn must return a vector or matrix with the first
    dimension being the length of the trajectory.
    '''                    
    reshaped_traj = tf_cat_traj_info(t_traj_info)
    ccs = t_single_traj_fn(reshaped_traj)
    mask = reshaped_traj['mask']
    if ccs.ndim == 2: 
        mask_reshape = mask.reshape((-1,1))        
        ccs = T.mean(ccs * mask_reshape, axis=0) #to avoid shallow copy
    else:
        ccs = T.mean(ccs * mask, axis=0)
    
    return ccs

def create_true_mean_function_nonseq(t_traj_info, t_single_traj_fn):
    '''        
    A faster version of _create_mean_function where t_single_traj_fn is non-sequential and therefore
    we can stack multiple trajectories as a single trajectory. 
    Note that t_single_traj_fn must return a vector or matrix with the first
    dimension being the length of the trajectory.
    '''                    
    reshaped_traj = tf_cat_traj_info(t_traj_info)
    ccs = t_single_traj_fn(reshaped_traj)
    mask = reshaped_traj['mask']
    starts = reshaped_traj['start_mark']
    if ccs.ndim == 2: 
        mask_reshape = mask.reshape((-1,1))        
        ccs = T.sum(ccs * mask_reshape, axis=0)/T.sum(starts) #to avoid shallow copy
    else:
        ccs = T.sum(ccs * mask, axis=0)/T.sum(starts)
    
    return ccs

def create_all_function_valid(t_traj_info, t_single_traj_fn, num_trajs=None, info_keys=None):
    '''        
    check if all checks are valid for each sequence
    '''  
    reshaped_traj = tf_cat_traj_info(t_traj_info)
    ccs = t_single_traj_fn(reshaped_traj)
    mask = reshaped_traj['mask']
    ccs = T.all(T.ge(ccs * mask,0.0), axis=0)
    return ccs                  

def _np2theano(name, np_arr):
    fn = [T.vector, T.matrix, T.tensor3][np_arr.ndim-1]
    return fn(name, dtype=np_arr.dtype)
            
# Note: This class assumes continuous actions
class NNPolicyUpdater(BasePolicyUpdater):
    '''
    Base class for policy updaters of continuous policies implemented by theano.
    '''
    def __init__(self, policy, **kwargs):        
        self.max_traj_len = kwargs.get('max_traj_length', -1)
        self.num_trajs = kwargs.get('num_trajs', -1)
        self._policy = policy
        self._params = self._policy.params
        self.gamma = kwargs.get('gamma', 0.98)
        baseline = kwargs.get('baseline', False)
        print ('using baseline:', baseline)
        self.clips = kwargs.get('clips', [])
        #if baseline is False or baseline is None: baseline = ZeroBaseline()  
        if isinstance(baseline, Baseline):
            self._baseline = baseline
        else:
            self._baseline = default_baselines[baseline](kwargs.get('past', 1), kwargs.get('fut', 1))
                
        self._traj_info_keys = None
        self._updater_built = False
        self._gamma_seq = np.array([self.gamma])                
        
    @property
    def policy(self):
        return self._policy
    
    @property
    def reactive_policy(self):
        return self._policy.reactive_policy
    
    def _save(self):
        params={}
        params['policy'] = self._policy._save()
        return params
        
    def _load(self, params):
        self._policy._load(params['policy'])
        return
                
    def _construct_traj_info(self, trajs):        
        '''
        Given a list of trajectories, return an ordered dictionary of vectors/matrices/tensors 
        storing trajectory information.                            
        
        The method stacks information from multiple trajectoreis in a higher-order structure.
        (e.g. stacks matrices into a tensor). To do this, trajectories are padded to
        have the same length. 
        '''
        
        N = len(trajs)        
        T = max(t.length for t in trajs)
        
        if self._gamma_seq.size < T: 
            self._gamma_seq = np.array([self.gamma**(i) for i in xrange(T)])

        act_dim = trajs[0].act.shape[1]
        # Note that 'state' here refers to the state of the filtering model using to provide input to the policy.
        # In case an ObservableModel is used, this is equivalent to observation.
        state_dim = trajs[0].states.shape[1]
                
        tensor_traj_X = np.zeros((N,T, state_dim))
        tensor_traj_pX = np.zeros((N,T, state_dim))
        tensor_traj_U = np.zeros((N,T, act_dim))
        tensor_traj_ctg = np.zeros((N,T))
        tensor_traj_mask = np.zeros((N,T))
        tensor_traj_len = np.zeros((N))
        
        for i in xrange(0, N):
            tensor_traj_X[i,:trajs[i].length,:] = np.copy(trajs[i].states)
            tensor_traj_pX[i,:trajs[i].length,:] = np.copy(trajs[i].prestates)
            tensor_traj_U[i,:trajs[i].length,:] = np.copy(trajs[i].act)

            rwd = np.copy(trajs[i].rewards)
            tmp_ctgs = np.array([-np.sum(rwd[j:]*self._gamma_seq[0:len(rwd[j:])]) for j in range(0,trajs[i].length)])
            tensor_traj_ctg[i,:trajs[i].length] = tmp_ctgs
            tensor_traj_mask[i,:trajs[i].length] = 1

        out = OrderedDict()           
        out['length'] = np.array([t.length for t in trajs])
        out['mask'] = tensor_traj_mask
        out['post_states'] = tensor_traj_X
        out['pre_states'] = tensor_traj_pX
        out['act'] = tensor_traj_U
        out['ctg'] = tensor_traj_ctg
        out['baseline'] = self._baseline.get_value_fn(trajs, out)
        out['advantage'] = out['ctg']-out['baseline']
        out['start_mark'] = np.zeros((N,T))
        out['start_mark'][:,0] = 1.0
    
        return out                       
                             
    def _build_updater(self, t_traj_info):
        '''
        This method is called on the first time update method is called.
        It is used to initialize any variables needed for executing updates.
        
        t_traj_info is an ordered dictionary of symbolic variables representing 
        trajectory information. This is the output of _symbolic_traj_processing
        method.
        '''
        raise NotImplementedError
        
    def gather_info(self, info, trajs):
        raw_rewards = np.array([np.sum(t.rewards) for t in trajs])
        info['reward_avg'] = np.mean(raw_rewards)
        info['reward_std'] = np.std(raw_rewards)
        info['fvel_avg'] = np.mean([np.sum(t.vel) for t in trajs])
        dbg_keys = trajs[0].dbg_info.keys()
        for k in dbg_keys:
            info[k] = np.mean([np.mean(t.dbg_info[k], axis=0) for t in trajs], axis=0)
        return info
        
    def update(self, trajs):
        traj_info = self._construct_traj_info(trajs)
        if not self._updater_built:            
            t_traj_info = OrderedDict([(k, _np2theano(k,v)) for k,v in traj_info.items()])
            self._build_updater(t_traj_info)
            self._updater_built = True
        info = self._update(traj_info)                            
        return self.gather_info(info, trajs)
     
    def _update(self, traj_info):
        '''
        Core update method: must return a (possibly empty) dictionary of values
        (used for monitoring)
        '''
        raise NotImplementedError


    
class GradientPolicyUpdater(NNPolicyUpdater):
    '''
    Updates policy by applying gradient descent on a cost function.
    '''
    def __init__(self, policy, max_traj_length, num_trajs, **kwargs):
        self._lr = kwargs['lr']
        self._optimizer = kwargs.pop('cg_opt','adam')
        self._normalize_grad = kwargs.pop('normalize_grad',False)
        NNPolicyUpdater.__init__(self, policy, **kwargs)

    def _t_single_traj_cost(self, t_single_traj_info):
        '''
        Computes the cost of a single trajectory. Returns a theano vector representing
        the cost for each time step.
        It is also possible to return multiple cost functions
        as a matrix where each row stores teh cost functions for a time step. 
        In this case _construct_updates and/or _build_updaters must be overriden
        to process cost functions correctly.
        
        t_single_traj_info is an ordered dictionary of symbolic variables
        storing information of a single trajectory.
        '''
        raise NotImplementedError
            
    def _t_cost(self, t_traj_info):        
        return create_true_mean_function_nonseq(t_traj_info, self._t_single_traj_cost)

    def _construct_updates(self, t_traj_info):
        '''
        t_traj_info is an ordered dictionary of symbolic variables
        storing information of trajectories.
        
        This function should return a tuple consisting of:
        1- A dictionary of theano updates.
        2- A dictionary of theano output variables (used for monitoring)        
        These are used by _build_updater to construct an update function          
        '''
        self._t_lr = theano.shared(self._lr, 'lr')

        t_cost = self._t_cost(t_traj_info)                                  
        beta = globalconfig.vars.args.beta
        grads, weight, updates = tf_get_normalized_grad_per_param(t_cost, self._policy.params, beta=beta, normalize=self._normalize_grad)
        opt_updates = optimizers[self._optimizer](0.0, self.policy.params, self._t_lr, all_grads = grads)   
        updates.extend(opt_updates)
    
        info = {'total_cost' : t_cost, 'var_g':T.sum([T.sum(gg**2) for gg in grads]), 'sum_g':T.sum([T.sum(T.abs_(gg)) for gg in grads])}
        if self._normalize_grad: info['gradient_weight'] = weight
        
        return updates, info
    
    def _build_updater(self, t_traj_info):
        t_updates, t_out = self._construct_updates(t_traj_info)            
        self._update_fn = theano.function(inputs=t_traj_info.values(),
                                          updates=t_updates, outputs=t_out.values(),
                                          on_unused_input='ignore')

        self._out_names = t_out.keys()

    def _update(self, traj_info):                
        out = self._update_fn(*traj_info.values())        
        return {k:v for (k,v) in zip(self._out_names, out)}

def t_vrpg_traj_cost(policy, t_single_traj_info):
    valid_len = t_single_traj_info['length']
    X = t_single_traj_info['pre_states'][:valid_len]
    U = t_single_traj_info['act'][:valid_len]
    adv = t_single_traj_info['advantage'][:valid_len]
    probs = policy._t_compute_prob(X,U)
    reinf_loss = T.log(probs+1e-13)*(adv)
    return reinf_loss 

class VRPGPolicyUpdater(GradientPolicyUpdater):  
    def _t_single_traj_cost(self, t_single_traj_info):
        return t_vrpg_traj_cost(self._policy, t_single_traj_info)
                    
##############################################################################
# Utility functions for TRPOPolicyUpdater 
def _t_gaussian_kl(t_old_mean, t_old_log_std, t_mean, t_log_std):    
    kl = (t_log_std-t_old_log_std) + \
         (T.exp(2*t_old_log_std) + T.square(t_old_mean-t_mean)) / (2 * T.exp(2*t_log_std)) - 0.5
         
    return T.sum(kl, axis=-1)   
        
##############################################################################


class TRPOPolicyUpdater(NNPolicyUpdater):
    '''
    Implements trust regionpolicy optimization.
    https://arxiv.org/abs/1502.05477
    '''

    def __init__(self, policy, **kwargs):
        NNPolicyUpdater.__init__(self, policy, **kwargs)
        self._step = kwargs['lr']
        self._params = policy.params
        X = T.tensor3()
        self._act_dist_fn = theano.function(inputs=[X], outputs=policy._t_compute_gaussian(X))
        self._opt = None
        self._t_prob_ratio_lims = [None, None]
        self._hvec = kwargs.pop('hvec', 'exact')

    def _append_actiondist_info(self, traj_info):
        traj_info['act_mean'], logstd = self._act_dist_fn(traj_info['pre_states'])
        N, T, d = traj_info['act_mean'].shape
        logstd = np.tile(logstd, (T, 1)).reshape((T, N, d)).transpose(1, 0, 2)
        traj_info['act_logstd'] = logstd

    def _construct_traj_info(self, trajs):
        out = NNPolicyUpdater._construct_traj_info(self, trajs)
        self._append_actiondist_info(out)
        return out

    def _t_traj_klscore(self, t_single_traj_info):
        valid_len = t_single_traj_info['length']
        X = t_single_traj_info['pre_states'][:valid_len]
        act_mean = t_single_traj_info['act_mean'][:valid_len]
        act_logstd = t_single_traj_info['act_logstd'][:valid_len]
        t_new_mean, t_new_logstd = self._policy._t_compute_gaussian(X)
        kl = _t_gaussian_kl(act_mean, act_logstd, t_new_mean, t_new_logstd)
        return kl

    def _t_append_actiondist_info(self, t_traj_info):
        X = t_traj_info['pre_states']
        N = X.shape[0]
        T = X.shape[1]

        X = X.reshape((N * T, -1))

        t_new_mean, t_new_logstd = self._policy._t_compute_gaussian(X)

        t_new_mean = t_new_mean.reshape((N, T, -1))
        t_new_logstd = t_new_logstd.reshape((N, T, -1))
        t_new_mean.name = 'new_act_mean'
        t_new_logstd.name = 'new_act_logstd'

        t_traj_info['new_act_mean'] = t_new_mean
        t_traj_info['new_act_logstd'] = t_new_logstd
        return t_traj_info

    def _t_ratio_limits(self, t_single_traj_info, ):
        r_max = globalconfig.vars.args.r_max
        r_min = 1 / float(r_max)

        prob_ratio = self._t_prob_ratio(t_single_traj_info)
        upper_bound_valid = T.lt(T.max(prob_ratio), r_max)
        lower_bound_valid = T.gt(T.min(prob_ratio), r_min)
        valid = T.switch(T.and_(lower_bound_valid, upper_bound_valid), 1, -1)
        return valid

    def _t_prob_ratio(self, t_single_traj_info):
        valid_len = t_single_traj_info['length']
        U = t_single_traj_info['act'][:valid_len]
        act_mean = t_single_traj_info['act_mean'][:valid_len]
        act_logstd = t_single_traj_info['act_logstd'][:valid_len]
        logprobs = -act_logstd - 0.5 * ((U - act_mean) ** 2) * np.exp(-2 * act_logstd)
        t_new_mean = t_single_traj_info['new_act_mean'][:valid_len]
        t_new_logstd = t_single_traj_info['new_act_logstd'][:valid_len]

        t_new_prec = T.exp(-2 * t_new_logstd)
        t_new_logprobs = -t_new_logstd - 0.5 * ((U - t_new_mean) ** 2) * t_new_prec
        prob_ratio = T.exp(T.sum(t_new_logprobs - logprobs, axis=1))
        return prob_ratio

    def _t_single_traj_cost(self, t_single_traj_info):
        valid_len = t_single_traj_info['length']
        adv = t_single_traj_info['advantage'][:valid_len]
        prob_ratio = self._t_prob_ratio(t_single_traj_info)

        t_new_logstd = t_single_traj_info['new_act_logstd'][:valid_len]
        t_new_prec = T.exp(-2 * t_new_logstd)

        cost = prob_ratio * (adv)
        cost = cost + 0.0 * T.sum(t_new_prec, axis=-1)
        return cost

    def _build_updater(self, t_traj_info, opt_inputs=None):
        if opt_inputs is None: opt_inputs = t_traj_info.values()
        t_traj_info = t_traj_info.copy()
        t_traj_info = self._t_append_actiondist_info(t_traj_info)

        print('Building Optimizer ...')
        opt_cost = create_true_mean_function_nonseq(t_traj_info, self._t_single_traj_cost)
        opt_constraint = create_true_mean_function_nonseq(t_traj_info, self._t_traj_klscore)
        ratio_checks = create_all_function_valid(t_traj_info, self._t_ratio_limits)
        gclip = globalconfig.vars.args.gclip
        ops = DefaultConstraintOptimizerOps(opt_cost, opt_constraint, opt_inputs,
                                            opt_inputs, self._params, ratio_checks,
                                            hvec=self._hvec)

        self._opt = ConstrainedOptimizer(ops, self._params, step=self._step)
        print ('Finished building optimizer')

    def _update(self, traj_info):
        self._opt.optimize(traj_info.values(), traj_info.values())
        reinf_cost = self._opt._ops.cost(*traj_info.values())
        return {'trpo_cost': reinf_cost}