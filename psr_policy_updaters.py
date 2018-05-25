#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 10:10:01 2017

@author: ahefny
"""

import numpy as np
import theano
import theano.compile
import theano.tensor as T
from theano.ifelse import ifelse
from time import time
import globalconfig

from collections import OrderedDict
from nn_policy_updaters import NNPolicyUpdater, GradientPolicyUpdater ,\
                               VRPGPolicyUpdater, TRPOPolicyUpdater, create_all_function_valid,\
                               create_true_mean_function_nonseq, tf_cat_traj_info, create_mean_function_nonseq,\
                               t_vrpg_traj_cost, tf_get_normalized_grad, tf_get_normalized_grad_per_param
from cg_optimizer import ConstrainedOptimizer, SemiAutoConstraintOptimizerOps, DefaultConstraintOptimizerOps
from SGD_opt import adam, RMSProp, adadelta, adagrad, optimizers
from psr_lite.psr_lite.utils.nn import dbg_print_shape, dbg_print

from IPython import embed
from psr_lite.psr_lite.utils.nn import dbg_print_shape, dbg_print_stats, \
dbg_print, dbg_nn_assert_notnan, test_theano_func, dbg_check_error, dbg_nn_assert,\
tf_get_normalized_grad, tf_get_normalized_grad_per_param, get_grad_update_old
from psr_lite.nn_diags import dbg_nn_raise_PredictionError, PredictionError, dbg_raise_BadPrediction

def _add_feats_to_traj_info(psrnet, traj_info):
    traj_info['obs'] = traj_info['post_states']
    X = traj_info['obs']
    traj_info['obs_feats'] = psrnet._process_obs(X.reshape((-1, X.shape[2]))).reshape((X.shape[0], X.shape[1], -1))
    U = traj_info['act']    
    traj_info['act_feats'] = psrnet._process_act(U.reshape((-1, U.shape[2]))).reshape((U.shape[0], U.shape[1], -1))
    
def _get_psr_single_trajinfo(psrnet, t_single_traj_info):
    '''
    Given a psr replace the 'pre_state' item in t_traj_info with psr prestates.
    The traj_info must contain obs_feats and act_feats for PSR filtering.
    '''
    valid_len = t_single_traj_info['length']
    UF = t_single_traj_info['act_feats'][:valid_len]
    XF = t_single_traj_info['obs_feats'][:valid_len]         
    H = psrnet.tf_compute_pre_states(XF,UF)    
    modified_traj = OrderedDict(t_single_traj_info.items())
    modified_traj['pre_states'] = H      
    
    return modified_traj    

def _get_psr_cat_trajinfo(psrnet, t_single_traj_info):
    '''
    Same as _get_psr_single_trajinfo but assumes the trajectory to be a concatentation
    of multiple trajectories.
    '''
    valid_len = t_single_traj_info['length']
    UF = t_single_traj_info['act_feats'][:valid_len]
    XF = t_single_traj_info['obs_feats'][:valid_len]         
    SM = t_single_traj_info['start_mark'][:valid_len]         
    h0 = psrnet.t_initial_state    
    
    def update_psr_state(o,a,sm,h):        
        h = ifelse(T.eq(sm, 0.0), h, h0)        
        hp1 = psrnet.tf_update_state(h,o,a)        
        return [hp1,h]

    H,_ = theano.scan(fn=update_psr_state,
                          outputs_info=[h0,None],
                          sequences=[XF,UF,SM])
    
    modified_traj = OrderedDict(t_single_traj_info.items())
    modified_traj['pre_states'] = H[1]    
    modified_traj['post_states'] = H[0]
    return modified_traj    

def _tf_get_psr_prestates_cat(psrnet, t_traj_info):    
    N,TT = t_traj_info['mask'].shape
    t_cat_traj_info = tf_cat_traj_info(t_traj_info)
    
    valid_len = t_cat_traj_info['length']
    UF = t_cat_traj_info['act_feats'][:valid_len]
    XF = t_cat_traj_info['obs_feats'][:valid_len]         
    SM = t_cat_traj_info['start_mark'][:valid_len]         
    h0 = psrnet.t_initial_state
    
    def update_psr_state(o,a,sm,h):        
        h = ifelse(T.eq(sm, 0.0), h, h0)        
        hp1 = psrnet.tf_update_state(h,o,a)        
        return [hp1,h]

    H,_ = theano.scan(fn=update_psr_state,
                          outputs_info=[h0,None],
                          sequences=[XF,UF,SM])
        
    states = H[1].reshape((N,TT,-1))
    states.name = 'psr_prestates'
        
    return states        
        
def _tf_get_psr_prestates_fixed_numtrajs(psrnet, t_traj_info, num_trajs):
    states = [None] * num_trajs
    for i in xrange(num_trajs):
        UF = t_traj_info['act_feats'][i]
        XF = t_traj_info['obs_feats'][i]
        #UF = dbg_print_shape('UF', UF)
        #XF = dbg_print_shape('XF', XF)
        states[i] = psrnet.tf_compute_pre_states(XF,UF)
        
    states = T.stack(states)
    states.name = 'psr_prestates'
    return states

def _tf_get_psr_prestates_batch(psrnet, t_traj_info):
    N,TT = t_traj_info['mask'].shape
    h0 = psrnet.t_initial_state
    H0 = T.tile(h0, (N,1))
    UF = t_traj_info['act_feats'][:,:-1,:].transpose(1,0,2)
    XF = t_traj_info['obs_feats'][:,:-1,:].transpose(1,0,2)
    
    fn_update = lambda o,a,h : psrnet.tf_update_state_batch(h,o,a)        
    H,_ = theano.scan(fn=fn_update, outputs_info=[H0], sequences=[XF,UF])
    H = T.concatenate([T.reshape(H0,(1,N,-1)), H],axis=0)
    return H.transpose(1,0,2)

def _tf_get_psr_prestates(psrnet, t_traj_info, num_trajs = None):    
    if not globalconfig.vars.args.dbg_nobatchpsr:  
        try:                        
            print 'Attempting to use batchified PSR filtering'               
            return _tf_get_psr_prestates_batch(psrnet, t_traj_info)                                         
        except:
            print 'WARNING: Could not used batchified PSR filtering'
    print 'num trajs is', num_trajs  
    if num_trajs > 0:
        # Use fixed num trajs for faster execution
        return _tf_get_psr_prestates_fixed_numtrajs(psrnet, t_traj_info, num_trajs)
    else:
        # Use concatenated trajectories            
        return _tf_get_psr_prestates_cat(psrnet, t_traj_info)

def get_grad_update( loss1, loss2, params, c1=1., c2=1., beta=0.1, normalize=True, clip_bounds=[], decay1=0.0, decay2=0.0):        
    combined_grads= []
    updates = []
    it = theano.shared(1.0,name='decay_iter::get_grad_update')
    g1, w1, u1 = tf_get_normalized_grad_per_param(loss1, params, beta=beta, normalize=normalize, clip_bounds=clip_bounds)
    g2, w2, u2 = tf_get_normalized_grad_per_param(loss2, params, beta=beta, normalize=normalize, clip_bounds=clip_bounds)
    
    updates.extend(u1)
    updates.extend(u2)
    for (gg1,gg2) in zip(g1,g2):
        combined_grad = gg1*c1*(1.-decay1)**it + gg2*c2*(1.-decay2)**it
        combined_grads.append(combined_grad)
    
    combined_loss = loss1*w1*c1*(1.-decay1)**it + loss2*w2*c2*(1.-decay2)**it
    
    updates.extend([(it, it+1)])
    results={'total_cost':combined_loss, 'cost2_avg':loss2*w2*c2*(1.-decay2)**it ,\
            'cost1_avg':loss1*w1*c1*(1.-decay1)**it, 'a1':w1,'a2':w2,\
            'updates':updates, 'grads':combined_grads,'params':params, 'total_grads':zip(g1,g2)}
    return results

def _tf_get_learning_rate(grads, beta=0.1):
        var = theano.shared(1.0, name='lr_g2')
        grad_sq = T.sum([T.sum(g**2) for g in grads])
        var_new = beta*var + (1.0-beta)*grad_sq    
        weight = 1.0 / T.sqrt(var_new)
        updates = [(var, var_new)] 
        return weight, updates 

                                      
class PSR_VRPGPolicyUpdater(VRPGPolicyUpdater):
    def __init__(self, *args, **kwargs):        
        self._beta_reinf = theano.shared(kwargs.pop('beta_reinf',1.0))
        self._beta_pred = theano.shared(kwargs.pop('beta_pred',1.0))
        self._grad_step = theano.shared(kwargs.pop('grad_step',1.0))
        self._beta_pred_decay = kwargs.pop('beta_pred_decay',1.0)        
        self._beta_only_reinf = 0.0 # TODO:remove once debug is done
        GradientPolicyUpdater.__init__(self, *args, **kwargs) 
        if globalconfig.vars.args.fix_psr:
            self._params = self._policy._policy.params   
        else:
            self._params = self._policy._psrnet.params + self._policy._policy.params
        self._vrpg_cost = lambda t: VRPGPolicyUpdater._t_single_traj_cost(self, t)

        # TODO: Now that we have normalization, should we include _proj_params 
        # in params ?        
        self._proj_params = self._policy._psrnet._params_proj        
        self._proj_step = self._policy._psrnet._opt_U
        
    def _construct_traj_info(self, trajs):
        out = VRPGPolicyUpdater._construct_traj_info(self, trajs)
        _add_feats_to_traj_info(self._policy._psrnet, out)                                
        return out
        
    def _t_single_traj_cost(self, t_single_traj_info):         
        return t_vrpg_traj_cost(self._policy, t_single_traj_info) 
    
    def _t_single_psr_cost(self, t_traj_info):
        return t_psr_pred_loss(self._policy._psrnet, t_traj_info) 
    
    def _t_psr_cost(self, t_traj_info):
        return create_true_mean_function_nonseq(t_traj_info, self._t_single_psr_cost)
    
                            
    def _construct_updates(self, t_traj_info):
        self._t_lr = theano.shared(self._lr, 'lr')
        t_traj_info = t_traj_info.copy()
        
        print 'Building PSR cost function ... ',        
        tic = time()   
        t_psr_traj_info = t_traj_info.copy()  
                
        t_psr_states = _tf_get_psr_prestates(self._policy._psrnet, t_psr_traj_info, self.num_trajs)
              
        t_psr_traj_info['pre_states'] = t_psr_states
                
        #t_costs = self._t_cost(t_psr_traj_info)  
        t_cost_reinf = self._t_cost(t_psr_traj_info)    #t_costs[0]
        t_cost_pred = self._t_psr_cost(t_psr_traj_info) #t_costs[1]
        if globalconfig.vars.args.dbg_prederror>0.0: 
            print 'checking pred error'
            t_cost_pred = dbg_raise_BadPrediction(t_cost_pred, 'bad prediction ')
        print 'finished in %f seconds' % (time() - tic)
        
        print 'Computing gradients ... normalize:',self._normalize_grad,
        tic = time()         
        gclip = globalconfig.vars.args.gclip
        beta = globalconfig.vars.args.beta
        decay1 =globalconfig.vars.args.decay1
        decay2 =globalconfig.vars.args.decay2
        results = get_grad_update(t_cost_pred,t_cost_reinf, self._params,
                                  self._beta_pred, self._beta_reinf, beta=beta,
                                  normalize=self._normalize_grad, clip_bounds=[-gclip,gclip], 
                                  decay1=decay1, decay2=decay2)
        
        updates = results['updates']
        t_grads = results['grads']
        keys=['cost1_avg', 'cost2_avg', 'total_cost', 'a1', 'a2']
        out = dict([(key,results[key]) for key in keys])
        out['reinf_loss'] = t_cost_reinf
        out['pred_loss'] = t_cost_pred
        
        out.update(self.policy._psrnet.tf_get_weight_projections(self.reactive_policy.params[0], t_psr_states))
        out.update({'var_g':T.sum([T.sum(gg**2) for gg in t_grads]), 'sum_g':T.sum([T.sum(T.abs_(gg)) for gg in t_grads])})
        
        print 'finished in {%f} seconds' % (time() - tic)
        beta_lr = globalconfig.vars.args.beta_lr
        lr = 1.0
        if beta_lr<>0.0:
            lr, lr_updates = _tf_get_learning_rate(t_grads, beta=beta_lr) #TODO: try with combined grads and original grads
            updates.extend(lr_updates)
            lr = dbg_print('lr', lr)
        print 'Computing optimizer updates ... ',
        tic = time()
        updates.extend(optimizers[self._optimizer](0.0, self._params, learning_rate=self._t_lr*lr, all_grads=t_grads))
        updates.extend([(self._t_lr, self._beta_pred_decay*self._t_lr)])
        print 'finished in %f seconds' % (time() - tic)
        return updates, out

    
    def _update(self, traj_info):
        try:
            out = self._update_fn(*traj_info.values())        
            return {k:v for (k,v) in zip(self._out_names, out)}
        except PredictionError: #if no valid update no op
            print 'Catch Prediction error do not update, '#,e.value
            return {} # Update PSR parameters
        
class PSR_AltOpt_TRPOPolicyUpdater(NNPolicyUpdater):
    def __init__(self, *args, **kwargs):        
        self._grad_step = theano.shared(kwargs.pop('grad_step', 1e-3), 'grad_step')
        self._lr = kwargs.pop('lr', 1e-3)
        self._beta_pred = kwargs.pop('beta_pred', 1.0)
        self._beta_reinf = kwargs.pop('beta_reinf', 0.0)
        self._beta_pred_decay = kwargs.pop('beta_pred_decay',1.0)     
        self._optimizer = kwargs.pop('cg_opt','adam')
        TRPO_method = kwargs.pop('TRPO_method', TRPOPolicyUpdater)
        super(PSR_AltOpt_TRPOPolicyUpdater, self).__init__(*args, **kwargs)
        self._beta_only_reinf = 0.0
        kwargs['lr'] = self._lr
        kwargs.pop('policy', None)
        trpo_args = (self._policy._policy,) + args[1:]        
        self._trpo = TRPO_method(*trpo_args, **kwargs)                                        
        self._normalize_grad = globalconfig.vars.args.norm_g
        XF = T.matrix()
        UF = T.matrix()
        H = self._policy._psrnet.tf_compute_pre_states(XF,UF)
        mu,S = self._policy._t_compute_gaussian(H)
        self._act_dist_fn = theano.function(inputs=[XF,UF], outputs=[mu,S])
        self._proj_step = self._policy._psrnet._opt_U
        
        self._policy_params = self._policy._policy.params
        
        if globalconfig.vars.args.fix_psr:
            self._params = [] 
        else:
            self._params = self._policy._psrnet.params #+ (self._policy._policy.params if self._beta_reinf<>0 else [])
        
        self._proj_params = self._policy._psrnet._params_proj
 
    def _construct_traj_info(self, trajs):
        out = NNPolicyUpdater._construct_traj_info(self, trajs)
        _add_feats_to_traj_info(self._policy._psrnet, out)
        out['act_mean'] = np.empty_like(out['act'])
        out['act_logstd'] = np.empty_like(out['act'])
        
        for i in xrange(len(out['length'])):
            out['act_mean'][i,:,:], out['act_logstd'][i,:,:] = \
                self._act_dist_fn(out['obs_feats'][i], out['act_feats'][i])
        return out
    
#    def psr_cost(self, t_single_traj_info):
#        return t_single_combined_cost(self._policy,t_single_traj_info)
    def _t_single_traj_cost(self, t_single_traj_info):
        return t_vrpg_traj_cost(self._policy, t_single_traj_info)
    
    def _t_single_psr_cost(self, t_traj_info):
        return t_psr_pred_loss(self._policy._psrnet, t_traj_info)
    
    def _t_psr_cost(self, t_traj_info):
        return create_true_mean_function_nonseq(t_traj_info, self._t_single_psr_cost)
   
    def _t_cost(self, t_traj_info):
        return create_true_mean_function_nonseq(t_traj_info, self._t_single_traj_cost)
 
    def _construct_updates(self, t_psr_traj_info):
        print 'Building PSR cost function ... ',      
        tic = time()
        #t_model_costs = create_true_mean_function_nonseq(t_psr_traj_info, self.psr_cost)                             
        print 'finished in %f seconds' % (time() - tic)
        t_cost_reinf = self._t_cost(t_psr_traj_info)    #t_costs[0]
        t_cost_pred = self._t_psr_cost(t_psr_traj_info) #t_costs[1]
        updates=[];
        if len( self._params)>0:
            #t_cost_pred = dbg_print_stats('predcost', t_cost_pred)  
            if globalconfig.vars.args.dbg_prederror>0.0:
                print 'checking pred error'
                t_cost_pred = dbg_raise_BadPrediction(t_cost_pred, 'bad prediction ')
            print 'finished in %f seconds' % (time() - tic)
            
            print 'Computing gradients ... normalize:',self._normalize_grad,
            tic = time()
            gclip = globalconfig.vars.args.gclip
            beta = globalconfig.vars.args.beta
            results = get_grad_update( t_cost_pred, t_cost_reinf, self._params,
                                       self._beta_pred, self._beta_reinf, beta=beta,
                                      normalize=self._normalize_grad, clip_bounds=[-gclip,gclip])
            
            updates = results['updates']
            t_grads = results['grads']
            print 'finished in %f seconds' % (time() - tic)
            beta_lr = globalconfig.vars.args.beta_lr
            lr = 1.0
            if beta_lr<>0.0:
                lr, lr_updates = _tf_get_learning_rate(t_grads, beta=beta_lr) #TODO: try with combined grads and original grads
                updates.extend(lr_updates)
                lr = dbg_print('lr', lr)
            
            print 'Computing optimizer updates ... ',
            tic = time()
            updates.extend(optimizers[self._optimizer](0.0, self._params, learning_rate=self._grad_step*lr, all_grads=t_grads))
            updates.extend([(self._grad_step, self._beta_pred_decay*self._grad_step)])
            print 'finished in %f seconds' % (time() - tic)
        return updates, {'reinf_loss' : t_cost_reinf, 'pred_loss' : t_cost_pred}
            
    def _build_updater(self, t_traj_info): 
        print 'Building TRPO Component'
        # Initialize reactive policy optimizer  
        t_psr_traj_info = t_traj_info.copy() 
        self._trpo._build_updater(t_traj_info)

        print 'Compiling state function ... ',
        tic = time()
        
        t_psr_states = _tf_get_psr_prestates(self._policy._psrnet, t_psr_traj_info, self.num_trajs)
        #t_psr_states = dbg_print_stats('psr_dbg_states: ', t_psr_states)         
        self._state_fn = theano.function(inputs=t_psr_traj_info.values(), outputs=t_psr_states, on_unused_input='ignore', \
                                         #mode=theano.compile.MonitorMode(post_func=test_theano_func)## REMOVE THIS LATER AFTER DEBUG
                                         )
        print 'finished in %f seconds' % (time() - tic)
        
        # Compute PSR parameter updates
        t_psr_traj_info['pre_states'] = t_psr_states
        updates, out = self._construct_updates(t_psr_traj_info)

        self._psr_update_fn = theano.function(inputs=t_traj_info.values(), updates=updates,\
                                               on_unused_input='ignore', outputs=out.values())
        self._out_names = out.keys()
            
        #t_traj_info
        print 'finished in %f seconds' % (time() - tic)
        
    def _update(self, traj_info):      
        try:
            obs = np.copy(traj_info['pre_states']) 
            # Replace observation model states with PSR states
            states = self._state_fn(*traj_info.values())
            traj_info['pre_states'] = states            
           
            # Update reactive policy
            out_trpo = self._trpo._update(traj_info)
                       
            # Update PSR Model
            traj_info['pre_states'] = obs            
            out = self._psr_update_fn(*traj_info.values())
            out = {k:v for (k,v) in zip(self._out_names, out)}
            out.update(out_trpo)
            return out
        except PredictionError:# as e: #if no valid update no op
            print 'Catch Prediction error do not update, '#,e.value
            return {}# Update PSR parameters

def t_psr_pred_loss(psrnet, t_single_traj_info):
    valid_len = t_single_traj_info['length']
    X = t_single_traj_info['obs'][:valid_len]
    U = t_single_traj_info['act_feats'][:valid_len]
    H = t_single_traj_info['pre_states'][:valid_len]        
    predictions = psrnet.tf_predict_obs(H,U)
    if globalconfig.vars.args.dbg_collapse:
        print 'checking collapse'
        predictions = dbg_nn_raise_PredictionError( predictions, 'trajectory is zero collapse!')

    #process_obs = psrnet.tf_extract_obs(X)
    process_obs = X
    #predictions = dbg_print_stats('pred', predictions)
    #process_obs = dbg_print_stats('true', process_obs)
    pred_cost = T.mean((predictions - process_obs) ** 2, axis=1)
    #pred_cost = (predictions - process_obs) ** 2
    pred_cost = dbg_print_stats('cost', pred_cost)
    return pred_cost

def t_single_combined_cost(policy, t_single_traj_info):
    pred_cost = t_psr_pred_loss(policy._psrnet, t_single_traj_info) 
    reinf_cost = t_vrpg_traj_cost(policy, t_single_traj_info)
    
    #pred_cost = dbg_print_stats('cost', pred_cost)
    #reinf_cost = dbg_print_stats('reinfcost', reinf_cost)
    comb_cost = T.stack([reinf_cost, pred_cost],axis=1).transpose()  
    #comb_cost = dbg_print_stats('comb', comb_cost)
    return comb_cost

# def t_psr_pred_cost(policy, t_single_traj_info):
#     X = t_single_traj_info
#     #X['pre_states'] = dbg_print_stats('psr_dbg_states: ', X['pre_states']) 
#     predict_obs = policy._psrnet.tf_predict_obs(X['pre_states'],X['act_feats'])  
#     if globalconfig.vars.args.dbg_collapse: 
#         print 'checking collapse' 
#         predict_obs = dbg_nn_raise_PredictionError( predict_obs, 'trajectory is zero collapse!')
#     process_obs = policy._psrnet.tf_extract_obs(X['obs'])
#     pred_cost = T.mean((predict_obs - process_obs) ** 2, axis=1) 
#     return pred_cost


class PSR_TRPOPolicyUpdater(TRPOPolicyUpdater):
    def __init__(self, *args, **kwargs):        
        self._b_reinf = kwargs.pop('beta_reinf',1.0)
        self._beta_reinf = theano.shared(self._b_reinf, borrow=True)
        self._b_pred = kwargs.pop('beta_pred',1.0)
        self._beta_pred = theano.shared(self._b_pred,borrow=True)
        self._beta_pred_decay = kwargs.pop('beta_pred_decay',1.0)
        self._beta_only_reinf = kwargs.pop('beta_only_reinf',0.0)
        
        TRPOPolicyUpdater.__init__(self, *args, **kwargs)
        self._proj_step = self._policy._psrnet._opt_U 
        
        self._policy_params = self._policy._policy.params if self._beta_only_reinf<>0 else []
        self._params = self._policy._psrnet.params + (self._policy._policy.params if self._beta_reinf<>0 else []) 
        
        self._proj_params = self._policy._psrnet._params_proj
        error_fn = lambda t: self._policy._psrnet.tf_1smse_wprestate(t['pre_states'],t['action_feats'],t['obs'])
        self._trpo_scores = lambda t: (TRPOPolicyUpdater._t_single_traj_cost(self, t), error_fn(t), self._t_traj_klscore(t))        
        XF = T.matrix()        
        UF = T.matrix()
        H = self._policy._psrnet.tf_compute_pre_states(XF,UF)
        mu,S = self._policy._t_compute_gaussian(H)
        self._act_dist_fn = theano.function(inputs=[XF,UF], outputs=[mu,S])
        
        self._opt = None
        
        self._current_w1s = theano.shared(self._policy._psrnet._t_W_1s.get_value())

    def _construct_traj_info(self, trajs):
        out = NNPolicyUpdater._construct_traj_info(self, trajs)                        
        _add_feats_to_traj_info(self._policy._psrnet, out) #prestates here are obs t-1
        
        out['act_mean'] = np.empty_like(out['act'])
        out['act_logstd'] = np.empty_like(out['act'])
        for i in xrange(len(out['length'])):
            out['act_mean'][i,:,:], out['act_logstd'][i,:,:] = \
                self._act_dist_fn(out['obs_feats'][i], out['act_feats'][i])
        
        return out        
                
    def _t_single_traj_cost(self, t_single_traj_info):        
        return self._t_single_traj_cost_combined(t_single_traj_info)

    def _t_single_traj_cost_combined(self, t_single_traj_info):
        reinf_loss = super(PSR_TRPOPolicyUpdater,self)._t_single_traj_cost( t_single_traj_info )
        model_cost = t_psr_pred_loss(self._policy._psrnet, t_single_traj_info)
        return  model_cost * self._beta_pred, reinf_loss* self._beta_reinf #, #reinf_loss
 
    def _t_single_traj_scores(self, t_single_traj_info):        
        cost, reinf = self._t_single_traj_cost_combined(t_single_traj_info)
        ws = (self._policy._psrnet._t_W_1s - self._current_w1s)
        kl = self._t_traj_klscore(t_single_traj_info) + self._step * T.sum(ws*ws)
        if kl.ndim<>reinf.ndim:
            kl = kl.reshape((kl.shape[0],1))
            embed() 
        return T.stack([cost, reinf, kl]).transpose()
        
                        
    def _build_updater(self, t_traj_info): 
        opt_inputs = t_traj_info.values()
        
        t_psr_states = _tf_get_psr_prestates(self._policy._psrnet, t_traj_info, self.num_trajs)
        t_psr_traj_info = t_traj_info.copy() 
        t_psr_traj_info['pre_states'] = t_psr_states
        t_psr_traj_info = self._t_append_actiondist_info(t_psr_traj_info) 
        
        t_opt_scores = create_true_mean_function_nonseq(t_psr_traj_info, self._t_single_traj_scores)
        opt_cost = t_opt_scores[0]
        reinf_cost = t_opt_scores[1]
        opt_constraint = t_opt_scores[2]
        ratio_lims = super(PSR_TRPOPolicyUpdater,self)._t_ratio_limits              
        ratio_checks = create_all_function_valid(t_psr_traj_info, ratio_lims)
        
        gclip = globalconfig.vars.args.gclip
        def optimizer(cost_fn, params, step):            
            ops = DefaultConstraintOptimizerOps(cost_fn, opt_constraint, opt_inputs,
                                                opt_inputs, params, ratio_checks,
                                                hvec = self._hvec)#, \
                                                #normalize=globalconfig.vars.args.norm_g, clip_bounds=[-gclip,gclip])
            
            return ConstrainedOptimizer(ops, params, step=step)
        
        #if self._beta_only_reinf>0.0:
        #    self._reinf_opt = optimizer(reinf_cost, self._policy_params, self._step*self._beta_only_reinf)
        #if self._proj_step>0:
        #    self._proj_opt = optimizer(opt_cost, self._proj_params, self._step*self._proj_step)
                            
        self._opt = optimizer([opt_cost,reinf_cost], self._params, self._step)
        
        return

                       
        
    def _update(self, traj_info):        
        try:
            #self._append_actiondist_info(traj_info)
            self._current_w1s.set_value( self._policy._psrnet._t_W_1s.get_value() )
            self._opt.optimize( traj_info.values(), traj_info.values() )
            #if self._proj_step>0:
            #    self._proj_opt.optimize( traj_info.values(),traj_info.values() )
            #if self._beta_only_reinf>0.0:
            #    self._reinf_opt.optimize( traj_info.values(),traj_info.values() )
            results = self.get_results(traj_info)
            self._beta_pred *= self._beta_pred_decay
            self._b_pred *= self._beta_pred_decay
        except PredictionError as e: #if no valid update no op
            print 'Catch Prediction error do not update, ',e.value
            return {}
        return results
    
    def get_results(self, traj_info):
        return {}
        
        opt_inputs =  traj_info.values()
        numtrajs = opt_inputs[0].shape[0]
        self._mean_cost = lambda cost_i_fn: sum([cost_i_fn(*[c[i] for c in opt_inputs]) for i in xrange(numtrajs)]) / float(numtrajs)
        self._std_cost = lambda cost_i_fn: np.sqrt(np.abs(sum([cost_i_fn(*[c[i]**2 for c in opt_inputs]) for i in xrange(numtrajs)]) / float(numtrajs)-self._mean_cost(cost_i_fn)**2)) 
        
        model_cost_mean = self._mean_cost( self._opt._ops._constraint_i)
        model_cost_std = self._mean_cost( self._opt._ops._constraint_i)
        reinf_cost_mean= 0; reinf_cost_std=0;
        if self._beta_only_reinf>0.0:
            reinf_cost_mean = self._mean_cost( self._reinf_opt._ops._cost_i)
            reinf_cost_std = self._mean_cost( self._reinf_opt._ops._cost_i)
        total_cost = self._mean_cost( self._opt._ops._cost_i)
        return {'model_cost_avg':model_cost_mean,'model_cost_std':model_cost_std,
                 'reinf_cost_avg':reinf_cost_mean,'reinf_cost_std':reinf_cost_std,
                 'total_cost':total_cost}
       







class jointOp_PolicyUpdater(object):
    def psr_pred_cost(self, t_single_traj_info):
        #pred_cost = t_psr_pred_cost(self._policy, t_single_traj_info)
        pred_cost = t_psr_pred_loss(self._policy._psrnet, t_single_traj_info)
        return pred_cost
    
    def _reinf_cost(self, t_single_traj_info):
        reinf_cost = t_vrpg_traj_cost(self._policy, t_single_traj_info)
        return reinf_cost

    def _construct_updates(self, t_traj_info):
        print 'Building PSR cost function ... ',        
        tic = time()   
        t_psr_states = _tf_get_psr_prestates(self._policy._psrnet, t_traj_info, self.num_trajs)
        t_psr_traj_info = t_traj_info.copy() 
        t_psr_traj_info['pre_states'] = t_psr_states  
        #t_pred_cost = T.sum(_create_mean_function(t_psr_traj_info, self.psr_pred_cost, num_trajs=self.num_trajs))                       
        #t_reinf_cost = T.sum(_create_mean_function(t_psr_traj_info, self._reinf_cost, num_trajs=self.num_trajs))        
        t_pred_cost = create_true_mean_function_nonseq(t_psr_traj_info, self.psr_pred_cost)                       
        t_reinf_cost = create_true_mean_function_nonseq(t_psr_traj_info, self._reinf_cost)    
        
        #t_reinf_cost = dbg_print_stats('reinfcost', t_reinf_cost)
        #t_pred_cost = dbg_print_stats('predcost', t_pred_cost)
                             
        print 'finished in %f seconds' % (time() - tic)
        print 'Get gradient function'
        tic = time()        
        gclip = globalconfig.vars.args.gclip
        t_grads = get_grad_update_old(t_pred_cost, t_reinf_cost, self._params, self._beta_pred, self._beta_reinf, clip_bounds=[-gclip,gclip])                      
        print 'finished in %f seconds' % (time() - tic)
        keys=['cost1_avg', 'cost2_avg', 'total_cost', 'a1', 'a2']
        out = dict([(key,t_grads[key]) for key in keys])
        
        out.update(self.policy._psrnet.tf_get_weight_projections(self.reactive_policy.params[0], t_psr_states))
        
        if globalconfig.vars.args.dbg_prederror>0.0: 
            print 'checking pred error'
            t_pred_cost = dbg_raise_BadPrediction(t_pred_cost, 'bad prediction ')
            
        print 'Compiling PSR update function ... ',
        tic = time()        
        psr_updates = optimizers[self._optimizer](0.0, self._params, learning_rate=self._grad_step, all_grads=t_grads['grads'])
        proj_updates = [] if self._proj_step==0.0 else optimizers[self._optimizer](t_pred_cost, self._proj_params, self._proj_step)
        reinf_updates = []
        if self._beta_only_reinf>0:
            print '\nlr ',self._lr, self._policy_params
            t_rgrads = get_grad_update_old(0.0, t_reinf_cost, self._policy_params, 0.0, 1.0)
            reinf_updates = optimizers[self._optimizer](0.0, self._policy_params, learning_rate=self._lr, all_grads=t_rgrads['grads'])
            
        print 'finished in %f seconds' % (time() - tic)
        return t_grads['updates'] + psr_updates + proj_updates + reinf_updates, out
   
   

class PSR_JointVRPG_PolicyUpdater(PSR_VRPGPolicyUpdater, jointOp_PolicyUpdater):
    def __init__(self,*args,**kwargs):
        self._beta_only_reinf = kwargs.pop('beta_only_reinf')
        PSR_VRPGPolicyUpdater.__init__(self, *args,**kwargs)
        
    #override
    def _construct_updates(self, t_psr_traj_info):
        return jointOp_PolicyUpdater._construct_updates(self, t_psr_traj_info)
 
        
class PSR_JointAltOp_PolicyUpdater(PSR_AltOpt_TRPOPolicyUpdater, jointOp_PolicyUpdater):
    def __init__(self, *args, **kwargs):
        return PSR_AltOpt_TRPOPolicyUpdater.__init__(self, *args, **kwargs)
    #override
    def _construct_updates(self, t_psr_traj_info):
        return jointOp_PolicyUpdater._construct_updates(self, t_psr_traj_info)
    
        
class NormVRPG_PolicyUpdater(VRPGPolicyUpdater, jointOp_PolicyUpdater):
    #override
    def _construct_updates(self, t_traj_info): 
        tic = time() 
        self._t_lr = theano.shared(self._lr, 'lr')
        t_reinf_cost = create_true_mean_function_nonseq(t_traj_info, self._reinf_cost)                         
        print 'finished in %f seconds' % (time() - tic)
    
        print 'Get gradient function'
        tic = time()        
        t_grads = get_grad_update(0.0, t_reinf_cost, self._params, 0.0, 1.0)                      
        print 'finished in %f seconds' % (time() - tic)
        keys=['cost2_avg', 'total_cost', 'a2']
        out = dict([(key,t_grads[key]) for key in keys])
        print 'Compiling PSR update function ... ',
        tic = time()
        updates = optimizers[self._optimizer](0.0, self._params, learning_rate=self._t_lr, all_grads=t_grads['grads'])
        
        print 'finished in %f seconds' % (time() - tic)
        return t_grads['updates'] + updates, out
       
