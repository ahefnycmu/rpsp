#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:13:25 2017

@author: ahefny
"""

import numpy as np
import theano
import theano.tensor as T
import policies
from psr_lite.utils.nn import dbg_print_shape
from IPython import embed
from nn_diags import PredictionError

class PSRLitePolicy(policies.BasePolicy):
    def __init__(self, psrnet, policy, init_a): 
        self._psrnet = psrnet
        self._policy = policy # Reactive policy
        
        self.params = self._psrnet.params + policy.params                
        
        self._init_af = psrnet._process_act(init_a)         
        self.output_dim = policy.output_dim
        #self._t_act = psrnet.t_afeat_mat
        #self._t_obs = psrnet.t_ofeat_mat
        
        # Note: At time t. t_ys is the policy given by all history up to t-1
        # We then execute t_act[t] and as a result we observe t_obs[t].
        # For a trajectorty of length T. These matroces should have T+1 rows.
        # t_act[0] should be set to a fake 'reset' action and t_obs[0] is the
        # reset observation. (the other possibility is to ignore the reset observation
        # and form matrices of T rows).
                        
        self.discrete = policy.discrete                
        
    @property
    def policy(self):
        return self._policy
    
    @property
    def reactive_policy(self):
        return self._policy
        
    def project(self, proj):
        self._policy.project(proj)
        self.params = self._psrnet.params + self._policy.params
        return
        
    def get_params(self):
        return np.hstack([self._psrnet.get_params(),self._policy.get_params()])
        
    def set_params(self, p):
        n = self._psr_param_dim
        if np.isnan(p).any() or np.isinf(p).any():
            print 'param is nan rffpsr policy! not updated'
            return
        assert not np.isinf(p).any(), 'param is inf rffpsr policy'
        self._psrnet.set_params(p[:n])
        self._policy.set_params(p[n:])
        
    #def _t_compute_states(self, Xf, Uf):
    #   TODO: Do we really need post states
    #    assert False
    #    return self._psrnet.tf_compute_post_states(Xf, Uf)
    
    # NOTE: This function recieves prestates
    def _t_compute_gaussian(self, H):
        return self._policy._t_compute_gaussian(X=H)
        
    # NOTE: This function recieves prestates
    def _t_compute_prob(self, H, U):                
        #t_ys = theano.scan(fn=self._policy._t_compute_prob_single,
        #                         outputs_info=None, sequences=h)
        #H = dbg_print_shape("compute_prob.state.shape=", H)
        if self.discrete:
            t_ys = self._policy._t_compute_prob(X=H)
            if U is not None:
                q = t_ys*U
                return T.sum(q, axis=1)
            else:
                return t_ys
        else:
            return self._policy._t_compute_prob(X=H,U=U)
        
    def reset(self):
        self._state = self._psrnet.initial_state
        self._act_f = self._init_af        
        
    def update_mem(self, xf):
        self._state = self._psrnet.update_state_wfeats(self._state, xf.squeeze(), self._act_f)
        #print self._state.shape, self._state
        
        if np.isnan(self._state).any():
            print 'ERROR:state is nan in update_mem::psrlite_policy'
            print self._state
            print 'init state', self._init_af, self._psrnet.initial_state
            raise PredictionError(self._state)
        
    def sample_action(self, x):
        xf = self._psrnet._process_obs(x)
        self.update_mem(xf)     
        act_n_prob = self._policy.sample_action(self._state)
                 
        a = act_n_prob[0]
        if type(a) is int:
            a = np.array([a])
        
        self._act_f = self._psrnet._process_act(a)
        return act_n_prob      
    
    def _load(self, params):
        print 'load psr policy'
        self._psrnet._load(params['rnnfilter'])
        self._policy._load(params['reactive'])
        self._init_af = params['init_af']
        self.reset()
        return
        
    def _save(self):
        params = {}
        params['rnnfilter'] = self._psrnet._save()
        params['reactive'] = self._policy._save()
        params['init_af'] = self._init_af
        return params  
    
                  

class RFFPSRNetworkPolicy(PSRLitePolicy):   
      
    
    def reset_psrnet(self, psrnet):
        psrnet._reset_psr(psrnet._rffpsr)
        self._psrnet = psrnet # RFFPSR_RNN
        self._fext_act = self._psrnet._fext_act
        self._fext_obs = self._psrnet._fext_obs
        self._psr_param_dim = len(self._psrnet.get_params())
        return
    

                