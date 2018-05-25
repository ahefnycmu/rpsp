#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 21:02:29 2017

@author: ahefny
"""

from __future__ import print_function
from utils.p3 import *

from time import time
import numpy as np
import theano
import theano.tensor as T
import theano.tensor.slinalg
import utils.nn
from psr_base import structtype
import psr_base
import feat_extractor
from noisy_model import NoisyModel
from IPython import embed
from utils.nn import dbg_print_shape

'''
A base class for recurrent filters.
The class assumes the following architecture:
    The network has access to a stream of actions a_t leading to observations a_t
    as well as guide input gi_t and guide output go_t signals that are used for training.
At time t:
    Non symbolic preprocessing:
    * o_t is processed to produce features ofeat_t
    * a_t is processed to produce features afeat_t
    
    Symbolic functions:
    * Given o_t,a_t,q_{t-1} the state q_t is inferred using _tf_update_state method.
    * Given q_t and a_{t+1}, o_{t+1} is inferred using tf_predict_obs method.
    * Given q_t and gi_{t+1}, go_{t+1} is inferred using tf_predict_guide method.            
    
Training is done by minimizing the prediction error of the guide signal. By default
gi_t = a_t and go_t = o_t but it can be changed to e.g. action and observation windows.
Guide signals are generated using _process_traj methods.
'''
class BaseRNNFilter(psr_base.ControlledModel):
    def __init__(self, state_dim, horizon, optimizer='sgd', optimizer_step=1.0,
                 optimizer_iterations=10, optimizer_min_step=1e-5, val_trajs=0, rng=None, opt_h0=False):                
        self._state_dim = state_dim
        self._horizon = horizon        
        self._optimizer = optimizer
        self._opt_step = optimizer_step
        self._opt_min_step = optimizer_min_step
        self._opt_iterations = optimizer_iterations
        self._val_trajs = val_trajs        
        self._h0_opt = opt_h0
        
        if rng is None:
            from utils.misc import get_default_rand 
            rng = get_default_rand()
            
        self.rng = rng
            
    def _process_traj(self, traj_obs, traj_act):
        return traj_obs, traj_act, traj_act, traj_obs
        
    def _process_obs(self, obs):
        return obs
        
    def _process_act(self, act):
        return act
    
    def tf_get_weight_projections(self, t_W0, t_psr_states, k='Wpred', dim=None):
        print (self.state_dimension)
        out = {}
        dim = self.state_dimension if dim is None else dim
        t_psrstates_reshaped = T.reshape(t_psr_states, (-1,dim)).T
        proj = T.dot(t_W0,t_psrstates_reshaped) 
        out[k] = T.stack([T.sum(T.var(proj,axis=1)), t_W0.norm(2)], axis=0)
        return out
        
    def _init_params(self, traj_obs, traj_act):
        self._t_state0 = None # Computable theano variable
        raise NotImplementedError
        
    def _build_graph(self, obs_dim, act_dim, on_unused_input='ignore'):        
        t_ofeat = T.vector('ofeat')
        t_afeat = T.vector('afeat')
        t_pre_state = T.vector('pre_state')
                
        t_post_state = self.tf_update_state(t_pre_state, t_ofeat, t_afeat)        
        t_pred_obs = self.tf_predict_obs(t_pre_state.reshape((1,-1)), t_afeat.reshape((1,-1))).reshape((-1,))        
                
        t_obs_mat = T.matrix('obs_mat')                                
        t_ofeat_mat = T.matrix('ofeat_mat')                                
        t_afeat_mat = T.matrix('afeat_mat')        
        t_guide_in_mat = T.matrix('guide_in_mat')
        t_guide_out_mat = T.matrix('guide_out_mat')                
        
        t_prestates_mat = self.tf_compute_pre_states(t_ofeat_mat, t_afeat_mat)
        t_prestates_mat.name = 't_prestates_mat'
        
        t_pred_err = T.sum((self.tf_predict_guide(t_prestates_mat, t_guide_in_mat) - t_guide_out_mat) ** 2)
        t_pred_err.name = 't_pred_err'
        
        opt_output = t_pred_err
        opt_inputs = [t_ofeat_mat,t_afeat_mat,t_guide_in_mat,t_guide_out_mat] 

        self._1smse = theano.function(inputs=[t_obs_mat, t_ofeat_mat, t_afeat_mat],
                                      outputs=self.tf_1smse_wprestate(t_prestates_mat, t_afeat_mat, t_obs_mat), 
                                      on_unused_input=on_unused_input, allow_input_downcast=True)

        self.update_state_wfeats = theano.function(inputs=[t_pre_state, t_ofeat, t_afeat],
                                          outputs=t_post_state,on_unused_input=on_unused_input,
                                          allow_input_downcast=True)
        
        self._pred_obs = theano.function(inputs=[t_pre_state, t_afeat],
                                        outputs=t_pred_obs,on_unused_input=on_unused_input)                 

        self._traj_pred_1s = theano.function(inputs=[t_ofeat_mat, t_afeat_mat],
                                             outputs=self.tf_predict_obs(t_prestates_mat, t_afeat_mat),
                                             on_unused_input=on_unused_input, allow_input_downcast=True)        


        
        self._get_pre_states = theano.function(inputs=[t_ofeat_mat, t_afeat_mat], 
                                          outputs=t_prestates_mat,
                                          on_unused_input=on_unused_input)
        #t_poststates_mat = self.tf_compute_post_states(t_ofeat_mat, t_afeat_mat)
        #t_poststates_mat.name = 't_poststates_mat'
        #self._get_post_states = theano.function(inputs=[t_ofeat_mat, t_afeat_mat], 
        #                                  outputs=t_poststates_mat,
        #                                  on_unused_input=on_unused_input)
        
        return opt_output, opt_inputs        
            
    def train(self, traj_obs, traj_act, traj_act_probs=None, on_unused_input='raise'):
        print('Building training graph ... ',end='')
        start = time()
        self._init_params(traj_obs, traj_act)
        opt_output, opt_inputs = self._build_graph(traj_obs[0].shape[1],traj_act[0].shape[1], on_unused_input=on_unused_input)        
        end = time()
        print('finished in %f seconds' % (end-start))         
               
        v = self._val_trajs
        if v < 0:
            assert v > -100
            v = len(traj_obs)*(-v)//100
        
        data = [self._process_traj(t[0],t[1]) for t in zip(traj_obs, traj_act)]  
        data = [list(d) for d in data if d is not None]
        
        if v == 0:
            train_data = data            
            val_data = []           
        else:
            train_data = data[:-v]            
            val_data = data[-v:]     
            print('total{} train size:{} validationsize:{}'.format(len(data), len(train_data),len(val_data)))                              
        print('Start training graph...')
        start = time()
        print (self._t_state0.get_value().sum())
        utils.nn.minimize_theano_fn(opt_output, opt_inputs,
                                    self._params_guide + self._common_params,
                                    num_samples=len(train_data),
                                    sampler=lambda i: train_data[i],
                                    num_val_samples=len(val_data),
                                    val_sampler=lambda i: val_data[i],
                                    initial_step=self._opt_step,
                                    min_step=self._opt_min_step,
                                    max_iterations=self._opt_iterations,
                                    optimizer=self._optimizer,
                                    on_unused_input=on_unused_input)
        
        self._state0 = self._t_state0.get_value()
        print ('state0', self._state0)
        end = time()
        print('finished in %f seconds'%(end-start))
    
    def update_state(self, state, obs, act):
        ofeat = self._process_obs(obs)
        afeat = self._process_act(act)
        return self.update_state_wfeats(state, ofeat, afeat)        
               
    def predict_obs(self, state, act):        
        return self._pred_obs(state, act)
    
    @property
    def state_dimension(self):
        return self._state_dim
        
    @property
    def horizon_length(self):
        return self._horizon

    @property
    def initial_state(self):
        return self.t_initial_state.eval()
    
    @property
    def t_initial_state(self):
        return self._t_state0
    
    def tf_update_state(self, t_state, t_ofeat, t_afeat):
        raise NotImplementedError
        
    def tf_update_state_batch(self, t_state_mat, t_ofeat_mat, t_afeat_mat):
        raise NotImplementedError
        
    def tf_predict_obs(self, t_prestates_mat, t_afeat_mat):
        raise NotImplementedError
     
    def tf_predict_guide(self, t_prestates_mat, t_guide_in_mat):
        return self.tf_predict_obs(t_prestates_mat, t_guide_in_mat)
    
    '''
    Compute the MSE of 1-step prediction of a trajectory given pre-states and
    action features.
    '''
    def tf_1smse_wprestate(self, t_prestates_mat, t_afeat_mat, t_obs_mat):         
        t_obsh_mat = self.tf_predict_obs(t_prestates_mat, t_afeat_mat)                                
        return T.mean(T.sum(((t_obsh_mat-t_obs_mat) ** 2),axis=1))
    
    '''
    Compute the SSE of 1-step prediction of a trajectory given pre-states and
    action features.
    '''
    def tf_1ssse_wprestate(self, t_prestates_mat, t_afeat_mat, t_obs_mat):
        t_obsh_mat = self.tf_predict_obs(t_prestates_mat, t_afeat_mat)
        return T.sum(((t_obsh_mat-t_obs_mat) ** 2))            
           
    '''
    Compute all states s.t. state[t] is after doing action[t]
    and then observing obs[t]
    '''
    def tf_compute_post_states(self, t_ofeat_mat, t_afeat_mat):
        # Use scan function
        hs,_ = theano.scan(fn=lambda o,a,h: self.tf_update_state(h,o,a),
                           outputs_info=self._t_state0,
                           sequences=[t_ofeat_mat,t_afeat_mat])
        return hs
        
    '''
    Compute all states s.t. state[t] is after doing action[t-1]
    and then observing obs[t-1]
    '''
    def tf_compute_pre_states(self, t_ofeat_mat, t_afeat_mat):
        hs = self.tf_compute_post_states(t_ofeat_mat[:-1],t_afeat_mat[:-1])        
        return T.concatenate([T.reshape(self._t_state0,(1,-1)), hs],axis=0)        

    def traj_1smse(self, traj_obs, traj_act):
        a_feat = self._process_act(traj_act)
        o_feat = self._process_obs(traj_obs)
        mse = self._1smse(traj_obs, o_feat, a_feat)
        assert not np.isnan(mse), 'traj 1smse is nan'  
        return mse
    
    def traj_predict_1s(self, traj_obs, traj_act):
        a_feat = self._process_act(traj_act)
        o_feat = self._process_obs(traj_obs)  
        return self._traj_pred_1s( o_feat, a_feat)    
        
    @property 
    def _common_params(self):
        return self._params_state + ( [self._t_state0] if self._h0_opt else [])  
        
    @property
    def params(self):

        return self._params_guide + self._common_params
    
    def tf_extract_obs(self, obs):
        return obs
    
    def _load(self, params):
        print ('load rnn_filter')
        for p in self.params:
            pval = params[p.name]
            p.set_value(pval)
        self._reset()
        return
        
    def _save(self):
        print ('save rnn_filter')
        params={}
        for p in self.params:
            params[p.name] = p.get_value()
        return params
     
    def _reset(self):
        pass
    

class ObsExtendedRNN(BaseRNNFilter):
    '''
    A class to extend the state of a recurrent filter with a window of
    previous observations.
    '''
    def __init__(self, base_model, obs_dim, win, mask):
        self._base_model = base_model
        self._obs_dim = obs_dim
        self._win = win            
        self._win_dim = self._obs_dim*self._win        
        self._t_mask = T.zeros(base_model.state_dimension) if mask else 1.0

            
    def _process_obs(self, obs):
        if obs.ndim==1:
            obs = obs.reshape(1,-1)
        last_obs = obs[:,-self._obs_dim:]
        ofeat = self._base_model._process_obs(last_obs)
        assert not np.isnan(ofeat).any(), 'obsfeat is not nan'
        assert not np.isinf(ofeat).any(), 'obsfeat is not inf'     
        new_obs = np.concatenate([ofeat.T , obs.T], axis=0).T
        return new_obs
       
    def _process_act(self, act):
        return self._base_model._process_act(act)
    
    def _process_traj(self, traj_obs, traj_act):
        if traj_obs.shape[0] <= self._fut + 3:
            return None
        else:
            data = psr_base.extract_timewins([traj_obs], [traj_act], self._fut, 1)[0]
            return self._base_model.process_obs(data.obs), \
                self._process_act(data.act), \
                self._process_fut_act(data.fut_act), \
                data.fut_obs
          

          
    def train(self, traj_obs, traj_act, traj_act_probs=None, on_unused_input='raise'):
        self._base_model.train(traj_obs, traj_act, traj_act_probs, on_unused_input)            
        #s0 = np.concatenate((self._base_model.initial_state, np.zeros(self._win_dim))) * self._mask        
        self._t_state0 =  T.concatenate([self._base_model.t_initial_state * self._t_mask, T.zeros(self._win_dim)],axis=0) 
     
        
        t_pre_state = T.vector('pre_state')
        t_ofeat = T.vector('ofeat')
        t_afeat = T.vector('afeat')
        t_post_state = self.tf_update_state(t_pre_state, t_ofeat, t_afeat)
                
        t_ofeat_mat = T.matrix('ofeat_mat')
        t_afeat_mat = T.matrix('afeat_mat')
        t_prestates_mat = self.tf_compute_pre_states(t_ofeat_mat, t_afeat_mat)
        
        self.update_state_wfeats = theano.function(inputs=[t_pre_state, t_ofeat, t_afeat],
                                              outputs=t_post_state,on_unused_input=on_unused_input,
                                              allow_input_downcast=True)
        
        self._traj_pred_1s = theano.function(inputs=[t_ofeat_mat, t_afeat_mat],
                                             outputs=self.tf_predict_obs(t_prestates_mat, t_afeat_mat),
                                             on_unused_input=on_unused_input, allow_input_downcast=True)
                                     
    
    @property
    def state_dimension(self):
        return self._base_model.state_dimension + self._win_dim
     
     
    def tf_get_weight_projections(self, t_W0, t_psr_states, k='Wfm'):
        W_base = t_W0[:,:-self._win_dim]
        W_win = t_W0[:,-self._win_dim:]
        t_psr_base_states = t_psr_states[:,:,:-self._win_dim]
        t_psr_win_states = t_psr_states[:,:,-self._win_dim:]
        
        out_base = self._base_model.tf_get_weight_projections(W_base, t_psr_base_states)
        out_win = self._base_model.tf_get_weight_projections(W_win, t_psr_win_states, k=k, dim=self._win_dim)
        out_base.update(out_win)
        return out_base 
    
    #@property
    #def t_initial_state(self):
    #    self._t_state0 = dbg_print_shape('tstate0::in::initialst', self._t_state0)
    #    return T.concatenate([self._t_state0, T.zeros(self._win_dim)], axis=0)

    def tf_update_state(self, t_state, t_ofeat, t_afeat):
        
        t_last_obs = t_ofeat[-self._obs_dim:]
        t_ofeat = t_ofeat[:-self._obs_dim]
        
        t_obswin = t_state[-self._win_dim:]
        t_state = t_state[:-self._win_dim]
                
        t_new_state = self._base_model.tf_update_state(t_state, t_ofeat, t_afeat)*self._t_mask         
        out = T.concatenate([t_new_state, t_obswin[self._obs_dim:], t_last_obs], axis=0)
        return out
    
#     def tf_compute_post_states(self, t_ofeat_mat, t_afeat_mat):
#         # Use scan function
#         state_0 = self.t_initial_state
#         
#         #t_ofeat_mat = dbg_print_shape('tofeatmat::post', t_ofeat_mat)
#         
#         #state_0 = dbg_print_shape('tf_post::s0', state_0)
#         hs,_ = theano.scan(fn=lambda fo,fa,h: self.tf_update_state(h,fo,fa),
#                            outputs_info=state_0,
#                            sequences=[t_ofeat_mat,t_afeat_mat])
#         return hs 
#     def tf_compute_pre_states(self, t_ofeat_mat, t_afeat_mat):
#         state_0 = self.t_initial_state #initial_state
#         #t_ofeat_mat = dbg_print_shape('tofeatmat', t_ofeat_mat)
#         hs = self.tf_compute_post_states(t_ofeat_mat[:-1],t_afeat_mat[:-1])        
#         return T.concatenate([T.reshape(state_0,(1,-1)), hs],axis=0)      
#
     
    def tf_update_state_batch(self, t_state_mat, t_ofeat_mat, t_afeat_mat):
        raise NotImplementedError
        t_last_obs_mat = t_ofeat_mat[:,-self._obs_dim:]
        t_ofeat_mat = t_ofeat_mat[:,:self._obs_dim]
        
        t_obswin_mat = t_state_mat[:,-self._win_dim]
        t_state_mat = t_state_mat[:,:self._win_dim]
                
        t_new_state_mat = self._base_model.tf_update_state_batch(t_state_mat, t_ofeat_mat, t_afeat_mat)                
        out = T.concatenate([t_new_state_mat, t_obswin_mat[:,self._obs_dim:], t_last_obs_mat], axis=1)
        return out                
    
    def tf_predict_obs(self, t_prestates_mat, t_afeat_mat):
        t_prestates_mat = t_prestates_mat[:,:-self._win_dim]
        return self._base_model.tf_predict_obs(t_prestates_mat, t_afeat_mat)
    
    def tf_predict_guide(self, t_prestates_mat, t_guide_in_mat):
        t_prestates_mat = t_prestates_mat[:,:-self._win_dim]
        return self._base_model.tf_predict_guide(t_prestates_mat, t_guide_in_mat)

    @property
    def params(self):
        return self._base_model.params
    
    def _load(self, params):
        self._base_model._load(params)
    
    def _save(self):
        return self._base_model._save()
    
    #TODO Remove this
    def get_projs(self):
        return None
    @property
    def _params_proj(self):
        return []
    @property
    def _opt_U(self):
        return 0.0
    
class RNNFilter(BaseRNNFilter):
    def __init__(self, state_dim, horizon, optimizer='sgd', optimizer_step=1.0,
                 optimizer_iterations=100, optimizer_min_step=1e-5, val_trajs=0,
                  rng=None):                
        BaseRNNFilter.__init__(self, state_dim, horizon, optimizer, optimizer_step,
                               optimizer_iterations, optimizer_min_step, val_trajs, rng=rng)
        
                                                     
    def _init_params(self, traj_obs, traj_act):
        d_o = traj_obs[0].shape[1]
        d_a = traj_act[0].shape[1]
        d_h = self._state_dim
        d_g = self._state_dim
                
        s0 = self.rng.rand(d_h)*0.5-0.25
        self._t_state0 = theano.shared(name='state0',value=s0)
        
        r = -4*np.sqrt(6.0/(d_o+d_a+d_h))                        
        self._t_Win_o = theano.shared(name='Win_o',
                                      value=self.rng.rand(d_o,d_h)*(2*r)-r)
        self._t_Win_a = theano.shared(name='Win_a',
                                      value=self.rng.rand(d_a,d_h)*(2*r)-r)
        
        r = -4*np.sqrt(6.0/(d_h+d_h))                        
        self._t_Wh = theano.shared(name='Wh',
                                   value=np.eye(d_h) + self.rng.randn(d_h,d_h)*0.01)                                   
        
        r = -4*np.sqrt(6.0/(d_h+d_g))
        self._t_W_hg = theano.shared(name='W_hg',
                                      value=self.rng.rand(d_h,d_g)*(2*r)-r)
        r = -4*np.sqrt(6.0/(d_a+d_g))
        self._t_W_ag = theano.shared(name='W_ag',
                                      value=self.rng.rand(d_a,d_g)*(2*r)-r)
        
        r = -4*np.sqrt(6.0/(d_h+d_o))
        self._t_W_out = theano.shared(name='W_out',
                                      value=self.rng.rand(d_g,d_o)*(2*r)-r)
        
        self._t_b_h = theano.shared(name='b_h', value=np.zeros(d_h))
        self._t_b_g = theano.shared(name='b_g', value=np.zeros(d_g))
        self._t_b_out = theano.shared(name='b_out', value=np.zeros(d_o)) 
         
        self._params_state = [self._t_Win_o,self._t_Win_a,self._t_Wh,self._t_b_h]                             
        self._params_obs = [self._t_W_hg,self._t_W_ag,self._t_W_out,self._t_b_g,self._t_b_out]
        self._params_guide = self._params_obs
                             
                            
    def tf_update_state(self, tin_hm1, tin_o, tin_a):
        tout_h = T.nnet.sigmoid(T.dot(tin_o,self._t_Win_o) +
                             T.dot(tin_a,self._t_Win_a) +
                             T.dot(tin_hm1,self._t_Wh) + self._t_b_h)
        
        return tout_h
            
    def tf_predict_obs(self, tin_h, tin_a):
        t_g = T.nnet.sigmoid(T.dot(tin_h, self._t_W_hg) + 
                          T.dot(tin_a, self._t_W_ag) + self._t_b_g)
        
        tout_o = T.dot(t_g, self._t_W_out) + self._t_b_out
        return tout_o
    
class ObservableRNNFilter(BaseRNNFilter):
    def __init__(self, model, horizon=1, optimizer_step=1.0,
                 optimizer_iterations=1000, optimizer_min_step=1e-5, val_trajs=0):
        self._model = model   
        BaseRNNFilter.__init__(self, model.state_dimension, horizon, optimizer_step,
                               optimizer_iterations, optimizer_min_step, val_trajs)
        s0 = model.initial_state       
        self._t_state0 = theano.shared(name='state0',value=s0)
        self._fext_obs = feat_extractor.FeatureExtractor()
        self._fext_act = feat_extractor.FeatureExtractor()
        self._fext_fut_act = feat_extractor.FeatureExtractor()
        
    #overwrite    
    def _init_params(self, traj_obs, traj_act):
        self._params_state = []
        self._params_obs = []
        self._params_guide = []  
        return
    
    def tf_compute_pre_states(self, t_ofeat_mat, t_afeat_mat):
        return self.tf_compute_post_states(t_ofeat_mat, t_afeat_mat)
    
    def tf_update_state(self, t_state, t_ofeat, t_afeat):
        return t_ofeat
    
    def tf_predict_obs(self, t_prestates_mat, t_afeat_mat):
        return t_prestates_mat
    
    def _tf_predict_horizon(self, t_prestates_mat, t_fafeat_mat):
        return t_prestates_mat    
    
    def get_params(self):
        return []
        
    def set_params(self, p):
        return
    
    @property
    def params(self):
        return []  
    
    def train(self, traj_obs, traj_act, traj_act_probs=None, on_unused_input='ignore'):
        print('Building obs training graph ... ',end='')
        start = time()
        self._init_params(traj_obs, traj_act)
        opt_output, opt_inputs = self._build_graph(traj_obs[0].shape[1],traj_act[0].shape[1], on_unused_input=on_unused_input)        
        end = time()
        print('finished in %f seconds' % (end-start))                 
        self._state0 = self._t_state0.get_value()
        return
      
class NoisyObservableRNNFilter(ObservableRNNFilter, NoisyModel):
    def __init__(self, model, horizon=1, optimizer_step=1.0,
                 optimizer_iterations=1000, optimizer_min_step=1e-5, val_trajs=0, 
                 obs_noise=0.0, obs_loc=0.0, state_noise=0.0, state_loc=0.0, rng=None):              
        ObservableRNNFilter.__init__(self, model, horizon, optimizer_step,
                               optimizer_iterations, optimizer_min_step, val_trajs)
        NoisyModel.__init__(self, obs_noise=obs_noise, obs_loc=obs_loc, 
                            state_noise=state_noise, state_loc=state_loc, \
                            state_dim=self._state_dim, rng=rng)
        
    def _process_traj(self, traj_obs, traj_act):
        return self._noisy_obs(traj_obs), traj_act, traj_act, self._noisy_obs(traj_obs)
        
    def _process_obs(self, obs):
        return self._noisy_obs( obs)    
  
    def tf_update_state(self, t_state, t_ofeat, t_afeat):
        ss = self._noisy_state(t_ofeat) 
        return ss


class RFFobs_RNN(ObservableRNNFilter):
    def __init__(self, model, fset=None, opt_U=0.0, opt_V=0.0, fut=1, dim=0, **kwargs):
        ObservableRNNFilter.__init__(self, model, **kwargs)
        self._fut = fut
        self._feature_set = fset
        self._past = 1
        self._opt_U = opt_U
        self._opt_V = opt_V
        self._f_obs = None
        self._f_act = None
        self._rffpsr = None
        #self._f_fut_act = None
        self._params_proj = []
        self.set_psr(None)
        s0 = np.zeros(dim)     
        self._t_state0 = theano.shared(name='state0',value=s0)
        return
    
    def extract_feats(self, traj_obs, traj_act):
        data,d = psr_base.extract_timewins(traj_obs, traj_act, self._fut, self._past)       
        print('Immediate')
        feats = structtype()
        self._fext_obs = self._feature_set['obs']
        feats.obs = self._fext_obs.build(data.obs).process(data.obs)
        self._fext_act =  self._feature_set['act']
        feats.act = self._fext_act.build(data.act).process(data.act)
        K = structtype()
        K.obs = feats.obs.shape[1]        
        K.act = feats.act.shape[1]
        self._feat_dim = K
        return
       
       
    def set_feat_set(self):
        self._fext_obs = self._feature_set['obs']
        self._fext_act = self._feature_set['act']
        K = structtype()
        K.obs = self._fext_obs._U.shape[1]
        K.act = self._fext_act._U.shape[1]
        self._feat_dim = K
        #self._fext_fut_act = self._feature_set['fut_act']
        #K.fut_act = self._fext_fut_act._U.shape[1]
        self.set_psr(None)
        return
  
    def _t_rff(self, x, V):
        y = T.dot(x, V)
        return T.concatenate([T.sin(y), T.cos(y)], axis=y.ndim-1) / T.sqrt(V.shape[1].astype(theano.config.floatX)) 
    
    '''
    Given an RFFPCA feature extractor return:        
        - A handle to an equivalent symbolic function.for vectors
        - A shared variable storing projection matrix.
        - A shared variable storing RFF matrix.
    '''
    def _t_rffpca(self, fext, name):
        U = theano.shared(name='U_%s' % name, value=fext._U.astype(theano.config.floatX))
        V = theano.shared(name='V_%s' % name, value=fext._base_extractor._V.astype(theano.config.floatX))
        f = lambda x: T.dot(self._t_rff(x, V), U)
        return f, U, V        
        
        
    def _reset_psr(self, dummy):
        self.set_feat_set()
        self.set_psr(dummy)
        return
        
    def set_psr(self, dummy):
        if self._opt_U>0.0 or self._opt_V>0.0:
            # Implement feature extraction using Theano
            if self._f_obs is None:
                # First time: create parameters
                self._f_obs, self._t_U_obs, self._t_V_obs = self._t_rffpca(self._fext_obs, 'obs')
                self._f_act, self._t_U_act, self._t_V_act = self._t_rffpca(self._fext_act, 'act')
                #self._f_fut_act, self._t_U_fut_act, self._t_V_fut_act = self._t_rffpca(self._fext_fut_act, 'fut_act')
            else:
                # Update parameters
                self._t_U_obs.set_value(self._fext_obs._U.astype(theano.config.floatX))
                self._t_V_obs.set_value(self._fext_obs._base_extractor._V.astype(theano.config.floatX))
                self._t_U_act.set_value(self._fext_act._U.astype(theano.config.floatX))
                self._t_V_act.set_value(self._fext_act._base_extractor._V.astype(theano.config.floatX))
                #self._t_U_fut_act.set_value(self._fext_fut_act._U.astype(theano.config.floatX))
                #self._t_V_fut_act.set_value(self._fext_fut_act._base_extractor._V.astype(theano.config.floatX))
        else:
            # Implement feature extraction using numpy
            self._f_obs = lambda x: x
            self._f_act = lambda x: x
            #self._f_fut_act = lambda x: x 
        
        
    
    def _init_params(self, traj_obs, traj_act):
        ObservableRNNFilter._init_params(self, traj_obs, traj_act)
        if self._opt_U>0.0:
            self._params_proj += [self._t_U_obs, self._t_U_act]#, self._t_U_fut_act]

        if self._opt_V>0.0:
            self._params_proj += [self._t_V_obs, self._t_V_act]#, self._t_V_fut_act]
        self._state_dim = self._f_obs(np.zeros(traj_obs[0].shape[1]))
        return
        
    def tf_update_state(self, t_state, t_ofeat, t_afeat):
        t_ofeat = self._f_obs(t_ofeat)
        return t_ofeat
         
    def _process_traj(self, traj_obs, traj_act):
        if traj_obs.shape[0] <= self._fut + 3:
            return None
        else:
            data = psr_base.extract_timewins([traj_obs], [traj_act], self._fut, 1)[0]
            return self._process_obs(data.obs), \
                self._process_act(data.act), \
                self._process_act(data.act), \
                data.fut_obs
                           
    def _process_obs(self, obs):
        ofeat = self._fext_obs.process(obs)
        assert not np.isnan(ofeat).any(), 'obsfeat is not nan'
        assert not np.isinf(ofeat).any(), 'obsfeat is not inf'     
        return ofeat
        
    def _process_act(self, act):
        afeat = self._fext_act.process(act)
        assert not np.isnan(afeat).any(), 'actfeat is not nan'
        assert not np.isinf(afeat).any(), 'actfeat is not inf'     
        return afeat
    
    def get_projs(self):
        projs={}
        if self._opt_U:
            projs['U_st'] = self._t_U_obs.get_value().T
            projs['U_act'] = self._t_U_act.get_value().T
        return projs
    
#     def _process_fut_act(self, fut_act):
#         futafeat = self._f_fut_act.process(fut_act)
#         assert not np.isnan(futafeat).any(), 'futafeat is not nan'
#         assert not np.isinf(futafeat).any(), 'futafeat is not inf'      
#         return futafeat
      
        
