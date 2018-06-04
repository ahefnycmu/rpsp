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
from utils.nn import reshape_mat_f
from utils.nn import cg_solve, cg_solve_batch, neumann_inv, neumann_inv_batch, batched_matrix_inverse
from utils.nn import CGD_optimizer
import rnn_filter

from collections import defaultdict
from feat_extractor import FeatureExtractor
import theano.printing
               
class RFFPSR_RNN(rnn_filter.BaseRNNFilter):        
    '''
    Theano wrapper of RFFPSR.
    '''   
    def __init__(self, psr, optimizer='sgd', optimizer_step=1.0,
                 optimizer_iterations=0, val_trajs=0, 
                 optimizer_min_step=1e-5, rng=None, opt_h0=False,
                 psr_norm='I', psr_cond='kbr', psr_iter=0, psr_smooth='I'):
        
        rnn_filter.BaseRNNFilter.__init__(self, psr.state_dimension, psr.horizon_length,
                                          optimizer, optimizer_step, optimizer_iterations,
                                          optimizer_min_step, val_trajs, rng=rng, opt_h0=opt_h0)
        self._psr_iter = psr_iter
        self._psr_cond = psr_cond
        self._state_norm = psr_norm   
        smooth_toks = psr_smooth.split('_')
        self._state_smooth = smooth_toks[0]
        if len(smooth_toks)>1:
            self._state_smooth_coeff = float(smooth_toks[1])
                
        self._f_obs = None
        self._f_act = None
        self._f_fut_act = None
        self._reset_psr(psr) 
        self._obs_dim = 0

        solve_dict = defaultdict(lambda: self._tf_solve_inverse, {'kbrcg': self._tf_solve_cg, 'kbrMIA': self._tf_solve_mia, 'I': self._tf_solve_ignore})
        solve_dict_batch = defaultdict(lambda: self._tf_solve_inverse_batch, {'kbrcg': self._tf_solve_cg_batch, 'kbrMIA': self._tf_solve_mia_batch, 'I': self._tf_solve_ignore})
        self._solve = solve_dict[self._psr_cond]
        self._solve_batch = solve_dict_batch[self._psr_cond]
        self._norm_method = defaultdict(lambda: self._t_state_noop , {'l2': self._t_state_l2norm, 
                                                                      'l2clamp': self._t_clamp_state_l2norm,
                                                                      'coord':self._t_clamp_state_coord})[self._state_norm]
        self._smooth = defaultdict(lambda: self._t_state_noop, {'interp': self._t_state_interpolate})[self._state_smooth]
        self._max_state_norm2 = 100.0
        self._max_state_norm = 10.0
        self._max_state_coord = 10.0
        self._min_state_coord = 1e-6
        
        
    def _t_rff(self, x, V):
        y = T.dot(x, V)
        return T.concatenate([T.sin(y), T.cos(y)], axis=y.ndim-1) / T.sqrt(V.shape[1].astype(theano.config.floatX))
                
    def _t_rffpca(self, fext, name):
        '''
        Given an RFFPCA feature extractor return:        
        - A handle to an equivalent symbolic function.for vectors
        - A shared variable storing projection matrix.
        - A shared variable storing RFF matrix.
        '''
        U = theano.shared(name='U_%s' % name, value=fext._U.astype(theano.config.floatX))
        V = theano.shared(name='V_%s' % name, value=fext._base_extractor._V.astype(theano.config.floatX))
        f = lambda x: T.dot(self._t_rff(x, V), U)
        
        return f, U, V        
        
    def set_psr(self, rff_psr):
        self._rffpsr = rff_psr
        self._fut = self._rffpsr._fut  
        self._feat_dim = self._rffpsr._feat_dim
        self._state_dim = self._rffpsr.state_dimension
        self._fext_fut_act = self._rffpsr._fext_fut_act
        self._fext_act = self._rffpsr._fext_act
        self._fext_obs = self._rffpsr._fext_obs
        self._feat_dim = self._rffpsr._feat_dim   
        return
        
    #overrides
    def _load(self, params):
        print 'load rffpsr rnn'
        self._rffpsr._load(params['rffpsr'])
        self._reset_psr(self._rffpsr)
        return
        
    #overrides
    def _save(self):
        params={}
        params['rffpsr'] = self._rffpsr._save()
        return params

    def _reset_psr(self, psr):
        self.set_psr(psr)
        self._f_obs = lambda x: x
        self._f_act = lambda x: x
        self._f_fut_act = lambda x: x
        return
            
    def train(self, traj_obs, traj_act, traj_act_probs=None, on_unused_input='raise'):
        self._reset_psr(self._rffpsr)
        return rnn_filter.BaseRNNFilter.train(self, traj_obs, traj_act, traj_act_probs=traj_act_probs, on_unused_input=on_unused_input)

    def _process_traj(self, traj_obs, traj_act):
        if traj_obs.shape[0] <= self._fut + 3:
            return None
        else:
            data = psr_base.extract_timewins([traj_obs], [traj_act], self._fut, 1)[0]
            return self._process_obs(data.obs), \
                self._process_act(data.act), \
                self._process_fut_act(data.fut_act), \
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
    
    def _process_fut_act(self, fut_act):
        futafeat = self._fext_fut_act.process(fut_act)
        assert not np.isnan(futafeat).any(), 'futafeat is not nan'
        assert not np.isinf(futafeat).any(), 'futafeat is not inf'      
        return futafeat
                                       
    def _init_params(self, traj_obs, traj_act):
        psr = self._rffpsr
        self._lambda = psr._lambda
        self._feat_dim = psr._feat_dim   
 
        self._t_W_s2ex = theano.shared(name='W_s2ex', value=psr._W_s2ex.astype(theano.config.floatX))
        self._t_W_s2oo = theano.shared(name='W_s2oo', value=psr._W_s2oo.astype(theano.config.floatX))
        self._t_W_h = theano.shared(name='W_h', value=psr._W_h.astype(theano.config.floatX))
        self._t_W_1s = theano.shared(name='W_1s', value=psr._W_1s.astype(theano.config.floatX))
        
        K = self._feat_dim
        self._t_UU_efa = theano.shared(name='UU_efa', value=psr._U_efa.T.reshape((-1, K.act), order='F').astype(theano.config.floatX)) 
        self._t_UU_efo = theano.shared(name='UU_efo', value=psr._U_efo.reshape((K.obs,-1), order='F').astype(theano.config.floatX))         
        self._t_U_oo = theano.shared(name='U_oo', value=psr._U_oo.astype(theano.config.floatX))
        self._t_UT_st = theano.shared(name='U_st', value=psr._U_st.T.astype(theano.config.floatX))

        s0 = psr.initial_state    
        print 'state0 : ', s0    
        self._t_state0 = theano.shared(name='state0',value=s0.astype(theano.config.floatX))
                           
        self._params_state = [self._t_W_s2ex,self._t_W_s2oo]
        self._params_obs = [self._t_W_1s]
        self._params_guide = [self._t_W_h]      

        t_prestates_mat = T.matrix()
        t_fa_mat = T.matrix()        
        
        self._pred_horizon = theano.function(inputs=[t_prestates_mat,t_fa_mat],
                                             outputs=self.tf_predict_guide(t_prestates_mat,t_fa_mat))                 
        return
    
    def get_projs(self):
        projs = self._rffpsr.get_projs()

        return projs
    
    def predict_horizon(self, state, fut_act):       
        fafeat = self._process_fut_act(fut_act.reshape(-1)).reshape(1,-1)
        o = self._pred_horizon(state.reshape((1,-1)), fafeat)     
        assert not np.isnan(o).any(), 'predict horizon is not nan'
        assert not np.isinf(o).any(), 'predict horizon is not inf'    
        return o.reshape((self._fut, -1))
                                             
    def get_params(self):
        return np.hstack([p.get_value().ravel() for p in self.params])
        
    def set_params(self, param_vec, check_before_update=False):
        i = 0
        if np.isnan(param_vec).any() or np.isinf(param_vec).any():
            print 'param is nan rffpsr policy! not updated'
            return
        if check_before_update:
            params_before = np.copy(self.get_params())

        for p in self.params:
            x = p.get_value(borrow=True)
            s = x.shape
            n = np.size(x)
            
            p.set_value(param_vec[i:i+n].reshape(s))
            i += n
        return
        
    def _tf_solve_inverse(self, A, b, reg):
        ''' solve via pseudo inverse Ax=b return x= inv(A).b'''
        A2 = T.dot(A.T, A)
        A2reg = A2 + T.eye(A.shape[1]) * reg
        vv = T.dot(b, A)
        v = T.dot(vv, matrix_inverse(A2reg))
        return v

    def _tf_solve_ignore(self, A, b, reg):
        return b
         
    def _tf_solve_cg(self, A, b, reg):
        A2 = T.dot(A.T, A)        
        vv = T.dot(b, A)
        v = cg_solve(A2, vv, iter=self._psr_iter, reg=reg)
        return v
    
    def _tf_solve_mia(self, A, b, reg):
        A2 = T.dot(A.T, A)        
        vv = T.dot(b, A)
        B = neumann_inv(A2, it=self._psr_iter, reg=reg)
        return T.dot(B, vv)

    def _tf_solve_batch_invalid(self, AA, B, reg):
        raise NotImplementedError
         
    def _tf_solve_inverse_batch(self, AA, B, reg):
        ''' solve via pseudo inverse Ax=b return x= inv(A).b'''
        N,d = B.shape
        AA2 = T.batched_dot(AA.transpose(0,2,1), AA)
        R = T.repeat(T.reshape(T.eye(d) * reg, (1,d,d)), N, axis=0)
        AA2reg = AA2 + R
        VV = T.batched_dot(B, AA)
        AAi = batched_matrix_inverse(AA2reg)
        V = T.batched_dot(VV, AAi)
        return V
        
    def _tf_solve_cg_batch(self, AA, B, reg):
        A2 = T.batched_dot(AA.transpose(0,2,1), AA)        
        VV = T.batched_dot(B, AA)
        V = cg_solve_batch(A2, VV, iter=self._psr_iter, reg=reg)
        return V

    def _tf_solve_mia_batch(self, AA, B, reg):
        A2 = T.batched_dot(AA.transpose(0,2,1), AA)                
        V = T.batched_dot(B, AA)
        B = neumann_inv_batch(A2, iter=self._psr_iter, reg=reg)
        return T.batched_dot(B, V)
                           
    def tf_update_state(self, t_state, t_obs, t_act):  
        t_ofeat = self._f_obs(t_obs)
        t_afeat = self._f_act(t_act)
                
        K = self._feat_dim          
        
        # Obtain extended state
        UU_efa = self._t_UU_efa    
        dot1 = T.dot(t_state, self._t_W_s2ex)
        dot1.name='tf_update_state::dot1'              
        C_ex = T.reshape(dot1,(K.exfut_obs, K.exfut_act))
        C_ex.name='tf_update_state::C_ex'                         
        
        # Condition on action
        B = reshape_mat_f(T.dot(UU_efa, t_afeat), (K.exfut_act, K.fut_act))
        B.name='tf_update_state::B'             
        C_efo_fa = T.dot(C_ex, B)
        C_efo_fa.name='tf_update_state::C_efo_fa'                

                
        # Obtain v = C_oo \ o_feat      
        C_oo_prj = T.dot(T.reshape(T.dot(t_state,self._t_W_s2oo), (K.oo, K.act)), t_afeat)
        C_oo_prj.name = 'tf_update_state::Cooprj'        
        C_oo = reshape_mat_f(T.dot(self._t_U_oo, C_oo_prj), (K.obs, K.obs))
        C_oo.name='tf_update_state::C_oo'
                
        v = self._solve(C_oo,t_ofeat, self._lambda['filter'])  
        v.name = 'tf_update_state::v'

        # Multply by v to condition on observation
        UU = self._t_UU_efo
        A = reshape_mat_f(T.dot(v, UU), (K.fut_obs, K.exfut_obs))
        A.name = 'tf_update_state::A'  
        ss = T.reshape(T.dot(A, C_efo_fa), [-1]) 
        ss.name = 'tf_update_state::ss_Cefodot'                                   
        ss = T.dot(self._t_UT_st, ss)
        ss.name = 'tf_update_state::Uss_dot'
        ss = self._norm_method(ss)
        ss = self._smooth(ss, t_state)

        self._dbg = lambda : None
        self._dbg.out = C_ex, C_oo, B, A, ss
        
        # Adding the sum of parameters fixes a Theano bug.
        return ss + sum(T.sum(p)*1e-30 for p in self.params)

    def _t_state_noop(self, state, *args):
        return state

    def _t_state_l2norm(self, state):
        ss_norm2 = T.sum(state**2)  
        state = T.switch(T.lt(ss_norm2 ,self._max_state_norm2),
                         state*(self._max_state_norm / T.sqrt(ss_norm2)),
                         state / T.sqrt(ss_norm2))
        return state

    def _t_clamp_state_l2norm(self, state):        
        ss_norm2 = T.sum(state**2)  
        state = T.switch(T.lt(ss_norm2 ,self._max_state_norm2),
                          state*(self._max_state_norm / T.sqrt(ss_norm2)),
                          state)
        return state
        
    def _t_clamp_state_coord(self, state):        
        return T.minimum(self._max_state_coord, T.maximum(self._min_state_coord, state))
           
    def _t_state_interpolate(self, state, prev_state):
        ''' convex interpolation with previous state to ensure smoothness'''
        interp = self._state_smooth_coeff
        #TODO: implement search direction and normalize
        state = (1.0-interp)*state + interp* prev_state
        return state

    def tf_update_state_batch(self, t_state_mat, t_obs_mat, t_act_mat):  
        t_ofeat_mat = self._f_obs(t_obs_mat)
        t_afeat_mat = self._f_act(t_act_mat)
        
        K = self._feat_dim          
        N = t_state_mat.shape[0]
        
        # Obtain extended state
        UU_efa = self._t_UU_efa                  
        C_ex = T.reshape(T.dot(t_state_mat, self._t_W_s2ex),(N, K.exfut_obs, K.exfut_act))
        C_ex.name='tf_update_state::C_ex'                 
        
        # Condition on action
        B = T.reshape(T.dot(t_afeat_mat, UU_efa.T), (N, K.fut_act, K.exfut_act)).transpose(0,2,1)
        B.name = 'tf_update_state::B'          
        #import pdb; pdb.set_trace()
        C_efo_fa = T.batched_dot(C_ex, B)
        C_efo_fa.name='tf_update_state::C_efo_fa'                
        
        # Obtain v = C_oo\o_feat                                
        C_oo_prj = T.batched_dot(T.reshape(T.dot(t_state_mat,self._t_W_s2oo), (N, K.oo, K.act)), t_afeat_mat)
        C_oo_prj.name = 'tf_update_state::Cooprj'
        C_oo = T.reshape(T.dot(C_oo_prj, self._t_U_oo.T), (N, K.obs, K.obs))
        C_oo.name='tf_update_state::C_oo'
                
        v = self._solve_batch(C_oo, t_ofeat_mat, self._lambda['filter'])                        
        v.name = 'tf_update_state::v'

        # Multply by v to condition on observation
        UU = self._t_UU_efo
        vproj = T.dot(v, UU)
        vproj.name ='tf_update_state::vproj'
        A = T.reshape(vproj,(N, K.exfut_obs, K.fut_obs)).transpose(0,2,1)
        
        A.name = 'tf_update_state::A'  
        ss = T.batched_dot(A, C_efo_fa).reshape([N,-1])        
        ss.name = 'tf_update_state::ss_Cefodot'                                   
        ss = T.dot(ss, self._t_UT_st.T)
        ss.name = 'tf_update_state::Uss_dot'
        ss = self._norm_method(ss)
        ss = self._smooth(ss, t_state_mat)
        
        self._dbg_batch = lambda : None
        self._dbg_batch.out = C_ex, C_oo, B, A, ss

        # Adding the sum of parameters fixes a Theano bug.
        return ss + sum(T.sum(p)*1e-30 for p in self.params)

    def tf_predict_obs(self, t_state, t_act):        
        is_vec = False
        if t_state.ndim == 1:
            is_vec = True
            t_state = t_state.reshape((1,-1))
            t_act = t_act.reshape((1,-1))
        
        t_obs = self._tf_predict_obs(t_state, t_act)
        
        if is_vec:
            t_obs = t_obs.reshape((1,-1))
            
        return t_obs
             
    def _tf_predict_obs(self, t_prestates_mat, t_act_mat):
        t_afeat_mat = self._f_act(t_act_mat)
        t_in = utils.nn.row_kr_product(t_prestates_mat, t_afeat_mat,
                                       name='_tf_predict_obs::t_in')    
        t_out = T.dot(t_in, self._t_W_1s) 
        t_out.name = '_tf_predict_obs::t_out'
        return t_out     
        
    def tf_predict_guide(self, t_prestates_mat, t_fa_mat):        
        t_fafeat_mat = self._f_fut_act(t_fa_mat)
        t_in = utils.nn.row_kr_product(t_prestates_mat, t_fafeat_mat)
        t_out = T.dot(t_in, self._t_W_h)
        return t_out 


class Extended_RFFPSR_RNN(RFFPSR_RNN):
    def __init__(self, *args, **kwargs):
        obs_dim = kwargs.pop('x_dim')
        win = kwargs.pop('win')
        super(Extended_RFFPSR_RNN, self).__init__(*args, **kwargs)
        self._obs_dim = obs_dim
        self._win = win
        self._win_dim = self._obs_dim * self._win

    def _process_obs(self, obs):
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        last_obs = obs[:, -self._obs_dim:]
        ofeat = self._fext_obs.process(last_obs)
        assert not np.isnan(ofeat).any(), 'obsfeat is not nan'
        assert not np.isinf(ofeat).any(), 'obsfeat is not inf'
        new_obs = np.concatenate([ofeat.T, obs.T], axis=0).T
        return new_obs

    def tf_extract_obs(self, obs):
        if obs.ndim == 2:
            last_obs = obs[:, -self._obs_dim:]
        else:
            last_obs = obs[-self._obs_dim:]
        return last_obs

    def _process_traj(self, traj_obs, traj_act):
        if traj_obs.shape[0] <= self._fut + 3:
            return None
        else:
            data = psr_base.extract_timewins([traj_obs], [traj_act], self._fut, 1)[0]
            return self._fext_obs.process(data.obs), \
                   self._process_act(data.act), \
                   self._process_fut_act(data.fut_act), \
                   data.fut_obs

    @property
    def state_dimension(self):
        return self._state_dim + self._win_dim

    @property
    def extended_dimension(self):
        return self._win_dim

    @property
    def initial_state(self):
        # return np.concatenate([self._t_state0.get_value(), np.zeros(self._win_dim)])
        return self.t_initial_state.eval()

    @property
    def t_initial_state(self):
        # return theano.shared(name='initstate0',value=self.initial_state.astype(theano.config.floatX))
        return T.concatenate([self._t_state0, T.zeros(self._win_dim)], axis=0)

    def tf_update_state(self, t_state, t_ofeat, t_afeat):
        t_obswin = t_ofeat[-self._win_dim:]
        t_state = super(Extended_RFFPSR_RNN, self).tf_update_state(t_state[:-self._win_dim], t_ofeat[:-self._win_dim],
                                                                   t_afeat)
        es = T.concatenate([t_state, t_obswin], axis=0)
        return es + sum(T.sum(p) * 1e-30 for p in (self.params + self._params_proj))

    def tf_update_state_batch(self, t_state_mat, t_ofeat_mat, t_afeat_mat):
        t_obswin_mat = t_ofeat_mat[:, -self._win_dim:]
        t_state_mat = super(Extended_RFFPSR_RNN, self).tf_update_state_batch(t_state_mat[:, :-self._win_dim],
                                                                             t_ofeat_mat[:, :-self._win_dim],
                                                                             t_afeat_mat)
        es = T.concatenate([t_state_mat, t_obswin_mat], axis=1)
        return es

    def tf_compute_post_states(self, t_ofeat_mat, t_afeat_mat):
        # Use scan function
        state_0 = self.t_initial_state
        hs, _ = theano.scan(fn=lambda fo, fa, h: self.tf_update_state(h, fo, fa),
                            outputs_info=state_0,
                            sequences=[t_ofeat_mat, t_afeat_mat])
        return hs

    def tf_compute_pre_states(self, t_ofeat_mat, t_afeat_mat):
        state_0 = self.t_initial_state  # initial_state
        hs = self.tf_compute_post_states(t_ofeat_mat[:-1], t_afeat_mat[:-1])
        return T.concatenate([T.reshape(state_0, (1, -1)), hs], axis=0)

    def _tf_predict_obs(self, t_extprestates_mat, t_act_mat):
        t_prestates_mat = t_extprestates_mat[:, :-self._win_dim]
        return super(Extended_RFFPSR_RNN, self)._tf_predict_obs(t_prestates_mat, t_act_mat)

    def tf_predict_guide(self, t_extprestates_mat, t_fa_mat):
        t_prestates_mat = t_extprestates_mat[:, :-self._win_dim]
        return super(Extended_RFFPSR_RNN, self).tf_predict_guide(t_prestates_mat, t_fa_mat)
