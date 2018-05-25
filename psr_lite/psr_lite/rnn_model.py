#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 12:38:44 2017

@author: ahefny
"""

import numpy as np
import theano
import theano.tensor as T
import theano.tensor.slinalg
from theano.tensor.nlinalg import matrix_inverse

import psr_base
import utils.nn
from psr_lite.psr_lite.utils.regression import ridge
from utils.nn import reshape_mat_f
from models import *
from IPython import embed
import psr_lite

 
 
 
class BaseTrainer(BatchTrainedFilteringModel):
    def __init__(self, batch_gen=TrainSetGenerator()):
        BatchTrainedFilteringModel.__init__(self, batch_gen=batch_gen)
        return
    
    def update(self, model, trajs):
        pass
 
             
'''
Theano wrapper of RFFPSR.
Note: 
    All symbolic functions (starting with tf) assume row ordering of trajectory
    matrices (row index is time).
'''
class RNN_trainer(BaseTrainer):         
    def __init__(self, batch_gen=TrainSetGenerator(), log=True, pi_update=True, reg=1e-4):
        BaseTrainer.__init__(self, batch_gen=batch_gen)
        self._iter = 0
        self._l2_lambda = reg
        self._log = log
        self._pi_update = pi_update
    
        
    def train(self, model, trajs):
        print 'Retraining #%d'%self._iter 
        
        psr = model.policy._psrnet._rffpsr
        rnn_model = model.policy._psrnet
        
        U_old = rnn_model.get_projs()    
        if self._pi_update=='reg':
            old_states = model.get_states(trajs)
        
        assert not np.isnan(trajs[0].obs).any(), 'obs is nan'
        assert not np.isinf(trajs[0].obs).any(), 'obs is inf'
        X_obs = [t.obs for t in trajs]
        X_act = [t.act for t in trajs]
        
        psr.train(X_obs, X_act, U_old=U_old)

        rnn_model = self.reset(model)
        rnn_model.train(X_obs, X_act)
        model.set_psr(rnn_model)
        
        assert rnn_model._feat_dim == psr._feat_dim, embed()
        U_new = rnn_model.get_projs()
        if self._log:
            from utils.log import Logger
            for (k,v) in U_new.items():
                d_min = min([U_new[k].shape[0], U_old[k].shape[0]])        
                Logger.instance().append('%s_%d' % (k, self._iter), np.dot(v[:d_min,:].T, U_old[k][:d_min,:]), print_out=self._log>1)
        #print 'GET new states'
        if self._pi_update=='proj':
            d_min = min([U_new['U_st'].shape[0], U_old['U_st'].shape[0]])
            UTUs = np.dot(U_old['U_st'][:d_min,:].T, U_new['U_st'][:d_min,:])
            self._update_policy_projection(UTUs, model)
        if self._pi_update=='reg':
            new_states = model.get_states(trajs)
            A = ridge(old_states, new_states, l2_lambda=self._l2_lambda)
            self._update_policy_projection(A, model)
            #if new_states.shape[1]<>old_states.shape[1]:
            #     embed()
        self._iter+=1
        assert model._psr == model.policy._psrnet, 'psrnet mismatch'
        assert model._psr == rnn_model, 'psrnet mismatch'
        assert model._psr._rffpsr == psr, 'psr mismatch'
        assert model._psr._feat_dim == model._psr._rffpsr._feat_dim, 'feat dim mismatch'
        print 'done retraining'
        return model
    
    def _update_policy_projection(self, A, model):
        model.policy.project(A)
        print 'policy update, Idiff= ', np.linalg.norm(A-np.eye(A.shape[0],A.shape[1]),ord=1)
        return
    
    def update(self, model, trajs):
        self._batch_gen.update(trajs)
        batch = self._batch_gen.gen_data()
        return self.train(model, batch)
    
    def reset(self, model):
        return self._reset_rnn(model)
    
    def _reset_rnn(self, model, delete=True):
        rnn_model = model.policy._psrnet
        psr = rnn_model._rffpsr
        if delete:
            print 'delete rnn'
            opt = rnn_model._optimizer 
            opt_step = rnn_model._opt_step
            opt_min_step = rnn_model._opt_min_step
            opt_iterations = rnn_model._opt_iterations 
            val_trajs = rnn_model._val_trajs 
            cond_iter = rnn_model._cg_iter
            opth0 = rnn_model._h0_opt
            opt_U  = rnn_model._opt_U
            opt_V  = rnn_model._opt_V
            rng = rnn_model.rng
            cl = rnn_model.__class__
            del rnn_model 
            rnn_model = cl( psr,cond_iter=cond_iter, rng=rng, 
                                  optimizer=opt, optimizer_step=opt_step,
                                  optimizer_iterations=opt_iterations, val_trajs=val_trajs, 
                                  optimizer_min_step=opt_min_step,
                                  opt_h0 = opth0, opt_U=opt_U, opt_V=opt_V )
        else:
            rnn_model.set_psr(psr)   
        return rnn_model
        
    