# -*- coding: utf-8 -*-
'''
Created on Tue Dec 20 19:48:50 2016

@author: ahefny
'''

from __future__ import print_function
from utils.p3 import *

import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla
from scipy.sparse.linalg import cg
from utils.nn import eval_CG, mia

import utils.linalg as ula
import utils.opt as opt
import utils.feats
import feat_extractor
from utils.regression import ridge 
from time import time
from psr_base import ControlledModel, UncontrolledModel, extract_timewins, structtype
from collections import defaultdict

from IPython import embed

        
def uniform_lambda(val):
    return {'s1a':val, 's1b':val, 's1c':val, 's1div':val,
            's2ex':val, 's2oo':val, 'filter':val, 'pred':val}
        
def uniform_blind_policy(data, feats):    
    n = data.obs.shape[1]
    return np.ones((1,n)), np.ones((1,n))

def gaussian_blind_policy(data, feats, l2_lambda):
    # Find the closest blind policy of the form
    # (a_{t+i} | history) ~ N(dot(w_i, h_t), \sigma_i^2)
    W_policy = ridge(feats.past, data.exfut_act, l2_lambda)
    r = np.dot(W_policy, feats.past) - data.exfut_act
    r2 = r*r
    S = np.mean(r*r,1).reshape((-1,1))
    
    blind_prob = np.sqrt(0.5/(np.pi * S)) * np.exp(-0.5*r2/S)
    
    d_a = data.act.shape[0]
    fut = data.fut_act.shape[0] / d_a
                
    blind_prop_future = np.prod(blind_prob[:d_a*fut,:], 0)
    blind_prop_extended = np.prod(blind_prob, 0)
            
    return blind_prop_future, blind_prop_extended

       
class RFFPSR(ControlledModel):
    def __init__(self, future_length, past_length, projection_dim=50,
                l2_lambda=1e-3, feature_set = None,
                s1_method='joint', past_projection=None,
                blind_policy = None, rng=None, psr_iter=0, psr_cond='kbr', psr_norm='I'):
                
        if rng is None:
            from utils.misc import get_default_rand 
            rng = get_default_rand()
        
        self.rng = rng
        if blind_policy is None:
            blind_policy = uniform_blind_policy
        
        if type(l2_lambda) is float:
            l2_lambda = uniform_lambda(l2_lambda)
        
        if feature_set is None:
            feature_set = feat_extractor.create_RFFPCA_featureset(1000, projection_dim, rng=self.rng)
         
        self._past_projection = past_projection
            
        self._fut = future_length
        self._past = past_length         
        self._p = projection_dim
        self._lambda = l2_lambda.copy()         
        self._feature_set = feature_set
        self._blind_policy = blind_policy
        self._clamp_method = None  if psr_norm=='I' else psr_norm      
        
        # For debugging purposes. User can set these fields instead of learning
        # projections from data.
        self._dbg_preset_U_efo = None
        self._dbg_preset_U_efa = None
        self._dbg_preset_U_oo = None
        self._dbg_preset_U_st = None
        self._frozen = False
        
        self._use_kbr = True
        self._psr_iter = psr_iter
        self._psr_cond = psr_cond
        
        s1_methods = {'joint' : self._s1_regression_joint,
                      'cond'  : self._s1_regression_cond,
                      'count'  : self._s1_regression_count,
                      'dist2dist' : self._s1_regression_dist2dist}
        
        if type(s1_method) is str:
            self._s1_regression = s1_methods[s1_method]
        else:
            self._s1_method = s1_method

        #self._solve = lambda A,b: eval_CG(np.abs(cond_iter))(A,b)
            
        solve_dict = defaultdict(lambda: self.solve_inverse_nopsd, {'kbrcg': self.solve_inv_cg, 'kbrMIA': self.solve_inv_mia, 'I': self.solve_ignore})    
        self._solve = solve_dict[self._psr_cond]
         
       
    def solve_inverse_nopsd(self, A, b, reg):
        return ula.reg_rdivide_nopsd(b, A, reg)
            
    def solve_inv_cg(self, A, b, reg):
        b = np.dot(b, A)
        d = A.shape[0]
        AA = np.dot(A.T,A)
        AA.ravel()[::d+1] += reg                    
        x,_ = cg(AA, b, maxiter=self._psr_iter)
        return x
    
    def solve_inv_mia(self, A, b, reg):
        A2 = np.dot(A.T, A)
        A2reg = A2 + np.eye(A.shape[1]) * reg
        vv = np.dot(b, A)
        v = mia(A2reg, vv, iter=self._psr_iter)
        return v
    
    def solve_inverse(self, A, b, reg):    
        vT = ula.reg_rdivide(np.dot(b, A), np.dot(A.T,A), reg)
        return vT
    
    def solve_ignore(self, A, b, reg):
        return b
    
    def initialize_random(self,  traj_obs, traj_act,traj_act_probs=None):
        print('Feature Extraction ... ', end='')
        start = time()
        data,d = self._extract_timewins(traj_obs, traj_act)         
        feats = self._extract_feats(data)  
        imp_weights = self._compute_importance_weights(data, feats, traj_act_probs)
        print('finished in %f seconds' % (time() - start))    
        print('finished in %f seconds' % (time() - start))
        K = self._feat_dim
        self._W_s2ex = self.rng.normal(size=(self._p, K.exfut_obs*K.exfut_act))
        self._W_s2oo = self.rng.normal(size=(self._p, K.oo*K.act))
        self._W_1s = self.rng.normal(size=(self._p*K.act, d.o))
        self._W_h = self.rng.normal(size=(self._p*K.fut_act, d.fo))
        self._state0 = self.rng.normal(size=(self._p))
        self._U_st = self.rng.normal(size=(K.exfut_obs*K.exfut_act,self._p))
        return
    
    def _load(self, params):        
        assert len(params)==15, 'wrong number of parameters (9): W_s2ex, W_s2oo, W_1s, W_h, state0,\
         U_st, U_efo, U_efa, U_oo, K, fext_past, fext_fut_obs, fext_fut_act, fext_obs, fext_act.'
        self._W_s2ex = params['W_s2ex']
        self._W_s2oo = params['W_s2oo']
        self._W_1s = params['W_1s']
        self._W_h = params['W_h']
        self._state0 = params['state0']
        self._U_st = params['U_st']
        self._U_efo = params['U_efo']
        self._U_efa = params['U_efa']
        self._U_oo = params['U_oo']
        self._feat_dim = structtype()
        self._feat_dim.__dict__ = params['K']
        for k,fext in self._feature_set.iteritems():
            fext.load(params['fext_'+k])
        self._fext_past = self._feature_set['past']
        self._fext_obs = self._feature_set['obs']
        self._fext_act =  self._feature_set['act']
        self._fext_fut_obs = self._feature_set['fut_obs']
        self._fext_fut_act = self._feature_set['fut_act']
        
    def _save(self):
        params = {}
        params['W_s2ex'] = self._W_s2ex
        params['W_s2oo'] = self._W_s2oo
        params['W_1s'] = self._W_1s
        params['W_h'] = self._W_h
        params['state0'] = self._state0
        params['U_st'] = self._U_st
        params['U_efo'] = self._U_efo
        params['U_efa'] = self._U_efa
        params['U_oo'] = self._U_oo
        params['K'] = self._feat_dim
        for k,fext in self._feature_set.iteritems():
            params['fext_'+k] = fext.save()
        return params
    
    def freeze(self):
        if not self._frozen:
            assert self._U_efo is not None, 'Uefo is None'
            self._dbg_preset_U_efo = np.copy(self._U_efo)
            assert self._U_efa is not None, 'Uefa is None'
            self._dbg_preset_U_efa = np.copy(self._U_efa)
            assert self._U_oo is not None, 'Uoo is None'
            self._dbg_preset_U_oo = np.copy(self._U_oo)
            assert self._U_st is not None, 'Ust is None'
            self._dbg_preset_U_st = np.copy(self._U_st)
            for k in self._feature_set.iterkeys():
                assert self._feature_set[k]._U is not None, 'U is None in feature set'
                self._feature_set[k]._frozen = True
                self._feature_set[k]._base_extractor._frozen = True
            self._frozen = True
        return
    
    def unfreeze(self):
        self._dbg_preset_U_efo = None
        self._dbg_preset_U_efa = None
        self._dbg_preset_U_oo = None
        self._dbg_preset_U_st = None
        self._frozen = False   
        for k in self._feature_set.iterkeys():
            self._feature_set[k]._frozen = False
            self._feature_set[k]._base_extractor._frozen = True
        return
    
    def train(self, traj_obs, traj_act, traj_act_probs=None, U_old=None):        
        print('Feature Extraction ... ', end='')
        start = time()
        data,d = self._extract_timewins(traj_obs, traj_act)         
        feats = self._extract_feats(data, U_old)        
        
        imp_weights = self._compute_importance_weights(data, feats, traj_act_probs)
        print('finished in %f seconds' % (time() - start))
                        
        print('2S Regression ... ', end='')
        start = time()
        # Note: s1_regression procudes pre-states 
        # (state[t] is after doing act[t-1] and observing obs[t-1])
        states, ex_states, im_states = self._s1_regression(data, feats, imp_weights)
        states = self._s2_regression(states, ex_states, im_states, U_old)
        
        print('State dimension = %d' % states.shape[1])
        print('finished in %f seconds' % (time() - start))
        
        print('Building Model ... ', end='')
        start = time()
        self._build_model(data, feats, states)
        print('finished in %f seconds' % (time() - start))
        return {'data':data, 'feats':feats, 'imp_weights':imp_weights, 'states':states,
                'ex_states':ex_states, 'im_states':im_states}
                
    def update_state(self, state, obs, act):                
        s = state
        a_feat = self._fext_act.process(act)        
        o_feat = self._fext_obs.process(obs)
        
        K = self._feat_dim
        
        # Obtain extended state
        C_ex = np.dot(s, self._W_s2ex).reshape((K.exfut_obs, K.exfut_act))
                
        # Condition on action
        B = np.dot(self._U_efa.T.reshape((-1, K.act), order='F'), a_feat).reshape((K.exfut_act, K.fut_act),order='F')
        C_efo_fa = np.dot(C_ex, B)
                
        # Obtain v = C_oo \ o_feat
        C_oo_prj = np.dot(np.dot(s,self._W_s2oo).reshape((K.oo, K.act)), a_feat)
        C_oo = np.dot(self._U_oo, C_oo_prj).reshape((K.obs, K.obs),order='F')
        vT = self._solve(C_oo, o_feat, self._lambda['filter'])                      
        
        # Multply by v to condition on observation
        UU = self._U_efo.reshape((K.obs,-1),order='F')
        A = np.dot(vT, UU).reshape((K.fut_obs, K.exfut_obs),order='F')
        ss = np.dot(A, C_efo_fa).reshape((-1,1))
        ss = np.dot(self._U_st.T, ss).ravel()
        
        # Clamp state
        if self._clamp_method is not None:
            ss = self._clamp_method(ss)
                
        return ss
                   
    def predict_shifted_state(self, state, act):
        raise 'Not implemented'
        
    def predict_obs(self, state, act):
        raise 'Not implemented'
        
    def predict_horizon(self, state, fut_act):  
        s = state
        a = fut_act.reshape(-1)
        a_feat = self._fext_fut_act.process(a)
        o = np.dot(np.kron(s, a_feat), self._W_h)
        return o.reshape((self._fut, -1))
        
    @property
    def state_dimension(self):
        return self._state0.size
        
    @property
    def horizon_length(self):
        return self._fut

    @property
    def initial_state(self):
        return self._state0
                   
    def set_state_clamp_method(self, method):
        if method is None:
            self._clamp_method = None
        elif isinstance(method,str):
            self._clamp_method = {'l2':self._clamp_state_l2norm,
                                  'coord':self._clamp_state_coord}[method]
        else:
            self._clamp_method = method
                          
    def _extract_timewins(self, traj_obs, traj_act):
        return extract_timewins(traj_obs, traj_act, self._fut, self._past)        
     
    def _extract_feats(self, data, U_old = None):                
        # Past
        print('Past')
        feats = structtype()
        self._fext_past = self._feature_set['past']
        feats.past = self._fext_past.build(data.past).process(data.past)        
        # Immediate
        print('Immediate')
        self._fext_obs = self._feature_set['obs']        
        feats.obs = self._fext_obs.build(data.obs).process(data.obs)        
        self._fext_act =  self._feature_set['act']
        feats.act = self._fext_act.build(data.act).process(data.act)
        # Future
        print('Future')
        self._fext_fut_obs = self._feature_set['fut_obs']
        feats.fut_obs = self._fext_fut_obs.build(data.fut_obs).process(data.fut_obs)
        self._fext_fut_act = self._feature_set['fut_act']
        feats.fut_act = self._fext_fut_act.build(data.fut_act).process(data.fut_act)
        
        '''
        Project future into subspace predictable by history        
        '''        
        if self._past_projection == 'svd':
            self._fext_fut_obs, feats.fut_obs = self._project_on_past(self._fext_fut_obs, feats.fut_obs, feats.past)
            self._fext_fut_act, feats.fut_act = self._project_on_past(self._fext_fut_act, feats.fut_act, feats.past)            
        else:
            assert self._past_projection is None, 'past projection is not None'
                                     
        # Shifted Future         
        print('Shifted Future')
        feats.shfut_obs = self._fext_fut_obs.process(data.shfut_obs)
        feats.shfut_act = self._fext_fut_act.process(data.shfut_act)                         
                
        print('Derived Features')
        # Derived Features:
        # Extended Future
        # Note that for exteded future observation, the current observation is
        # the "lower order" factor. This makes filtering easier.        
        feats.exfut_obs = ula.khatri_rao_rowwise(feats.shfut_obs, feats.obs)
        feats.exfut_act = ula.khatri_rao_rowwise(feats.act, feats.shfut_act)  
                        
        if self._dbg_preset_U_efo is None:
            self._U_efo,_,exfut_obs = ula.rand_svd_f(feats.exfut_obs.T, k=self._p, rng=self.rng)
            feats.exfut_obs, self._U_efo = self._keep_proj_sign(self._U_efo, 'UU_efo', feats.exfut_obs, exfut_obs, U_old) 
        else:
            self._U_efo = self._dbg_preset_U_efo
            feats.exfut_obs = np.dot(feats.exfut_obs, self._U_efo)
            
        if self._dbg_preset_U_efa is None:
            self._U_efa,_,exfut_act = ula.rand_svd_f(feats.exfut_act.T, k=self._p, rng=self.rng)
            feats.exfut_act, self._U_efa = self._keep_proj_sign(self._U_efa, 'UU_efa', feats.exfut_act, exfut_act, U_old) 
        else:
            self._U_efa = self._dbg_preset_U_efa
            feats.exfut_act = np.dot(feats.exfut_act, self._U_efa)
        
        # Observation Covariance
        feats.oo = ula.khatri_rao_rowwise(feats.obs, feats.obs)        
        
        if self._dbg_preset_U_oo is None:                        
            # Project lower triangle part of observation covariance            
            d_obs = feats.obs.shape[1]
            lt_idx = np.array([i for i in xrange(d_obs**2) if (i//d_obs) >= (i%d_obs)])
            ut_idx = (lt_idx%d_obs)*d_obs + lt_idx//d_obs

            feats.oo = feats.oo[:,lt_idx]            
            self._U_oo,_,oo = ula.rand_svd_f(feats.oo.T, k=self._p, rng=self.rng)
            feats.oo, self._U_oo = self._keep_proj_sign(self._U_oo, 'U_oo', feats.oo, oo, U_old) 
            # Convert singular vectors back to represent symmetric matrices
            Usym = np.empty((d_obs**2, self._U_oo.shape[1]))
            Usym[lt_idx,:] = self._U_oo
            Usym[ut_idx,:] = self._U_oo
            self._U_oo = Usym
            
            
        else:
            self._U_oo = self._dbg_preset_U_oo
            feats.oo = np.dot(feats.oo, self._U_oo)                
                            
        K = structtype()
        K.obs = feats.obs.shape[1]        
        K.act = feats.act.shape[1]
        K.past = feats.past.shape[1]
        K.fut_obs = feats.fut_obs.shape[1]
        K.fut_act = feats.fut_act.shape[1]
        K.exfut_obs = feats.exfut_obs.shape[1]
        K.exfut_act = feats.exfut_act.shape[1]
        K.oo = feats.oo.shape[1]
                        
        self._feat_dim = K
                
        for (k,v) in K.__dict__.items():
            print('%s:%d' % (k,v))
                    
        return feats
     
    def _keep_proj_sign(self, Unew, k, feats, ufx, U_old):
        if U_old is not None:
            dmin = min([Unew.shape[0],U_old[k].shape[0]])
            signs =  np.diag(np.dot(Unew[:dmin,:].T, U_old[k][:dmin,:]))
            dout = signs.shape[0]
            Unew[:,:dout] = Unew[:,:dout] * signs
            new_feats = np.dot(feats, Unew)
            #print ('UTU_%s'%k, np.dot(Unew[:dmin,:].T,  U_old[k][:dmin,:]) )
            assert (np.diag(np.dot(Unew[:dmin,:].T,  U_old[k][:dmin,:]))>=0).all(), 'flipped sign %s'%k
        else:
            new_feats = ufx.T
        return new_feats, Unew
    
    def get_projs(self):
        U = {}
        U['U_st'] = np.copy(self._U_st)
        U['U_efo'] = np.copy(self._fext_fut_obs._U)
        U['U_fut_act'] = np.copy(self._fext_fut_act._U)
        U['U_obs'] = np.copy( self._fext_obs._U)
        U['U_act'] = np.copy( self._fext_act._U)
        U['UU_efo'] = np.copy( self._U_efo)
        U['UU_efa'] = np.copy( self._U_efa)
        U['U_oo'] = np.copy(self._U_oo)
        return U
     
    def _project_on_past(self, fext_fut, fut, past):
        if fut.shape[1] > self._p:
            C_f_p = np.dot(fut.T, past)
            U_f = spla.svd(C_f_p, full_matrices=False, compute_uv=True, overwrite_a=True)[0]
            U_f = U_f[:,:self._p]
            fext_fut = feat_extractor.ProjectionFeatureExtractor(fext_fut, U_f)
            fut = np.dot(fut, U_f)
        
        return fext_fut, fut
        
    def _compute_importance_weights(self, data, feats, traj_act_probs):                        
        if traj_act_probs is None:
            return None, None
        else:
            bounds = (self._past, self._fut)
            fut_extractor = utils.feats.finite_future_feat_extractor(self._fut)
            extended_extractor = utils.feats.finite_future_feat_extractor(self._fut+1)  
            traj_act_probs = [t.reshape((-1,1)) for t in traj_act_probs]

            # Compute the probability of action sequences given the non-blind policy            
            prob_future = np.prod(utils.feats.flatten_features(traj_act_probs, fut_extractor, bounds)[0], 0)             
            prob_extended = np.prod(utils.feats.flatten_features(traj_act_probs, extended_extractor, bounds)[0], 0)             
            
            blind_prop_future, blind_prop_extended = self._blind_policy(data, feats)
                                                            
            weights_future = blind_prop_future / prob_future
            weights_extended = blind_prop_extended / prob_extended
            
            return weights_future, weights_extended
                                
    '''
    Given data matrices A,B,C, compute the conditional expectation operator of A
    conditioned on B given C. 
    
    The method returns a matrix D [A.shape[1] x B.shape[1]] such that 
    M=D[i,:].reshape((A.shape[1],B.shape[1])) is a matrix satisfying
    E[A|B=b; C=C[i,:]] = Mb
    
    The second return is the regression weight matrix used to compute D
    '''    
    def _estimate_condop(self, A, B, C, reg_lambda, div_lambda, importance_weights):
        N,da = A.shape
        db = B.shape[1]
        dab = da*db
        dbb = db*db                
        
        reg_out = np.empty((N,dab+dbb))
        reg_out[:,:dab] = ula.khatri_rao_rowwise(A,B)
        reg_out[:,dab:] = ula.khatri_rao_rowwise(B,B)
        
        W = ridge(C, reg_out, reg_lambda, importance_weights)
        
        est_reg_out = np.dot(C,W)        
        output = np.empty((N,dab))
        
        for i in xrange(N):
            C_ab = est_reg_out[i,:dab].reshape((da,db))
            C_bb = est_reg_out[i,dab:].reshape((db,db))
            #C_a_b = self._solve(C_bb,C_ab,div_lambda)
            C_a_b = ula.reg_rdivide_nopsd(C_ab, C_bb, div_lambda)
            output[i,:] = C_a_b.reshape(-1)
                
        return output,W
      
    def _s2_regression(self, states, ex_states, im_states, U_old):                
        print('State Projection')
        if self._dbg_preset_U_st is None:
            self._U_st,_,states_fx = ula.rand_svd_f(states.T, k=self._p, rng=self.rng)                      
            states, self._U_st = self._keep_proj_sign(self._U_st, 'U_st', states, states_fx, U_old) 
        else:
            self._U_st = self._dbg_preset_U_st
            states = np.dot(states, self._U_st)
            
        self._feat_dim.state = states.shape[1]

        print('Stage 2 Regression')
        self._W_s2ex = ridge(states, ex_states, self._lambda['s2ex'])
        self._W_s2oo = ridge(states, im_states, self._lambda['s2oo'])
        
        return states
                            
    def _s1_regression_joint(self, data, feats, imp_weights):        
        print('Stage 1 Regression')
        states,self._dbg_W_s1a = self._estimate_condop(
            feats.fut_obs, feats.fut_act, feats.past,
            self._lambda['s1a'], self._lambda['s1div'], imp_weights[0])
        ex_states,self._dbg_W_s1b = self._estimate_condop(
            feats.exfut_obs, feats.exfut_act, feats.past,
            self._lambda['s1b'], self._lambda['s1div'], imp_weights[1])
        im_states,self._dbg_W_s1c = self._estimate_condop(
            feats.oo, feats.act, feats.past,
            self._lambda['s1c'], self._lambda['s1div'], None)
                        
        return states, ex_states, im_states
                                     
    def _s1_regression_cond(self, data, feats, imp_weights):
        print('Stage 1A Regression')
        s1a_in = ula.khatri_rao_rowwise(feats.past, feats.fut_act)
        s1a_out = feats.fut_obs
        W_s1a = ridge(s1a_in, s1a_out, self._lambda['s1a'], imp_weights[0])
        #W_s1a = W_s1a.reshape((self._feat_dim.past, -1))
        W_s1a = W_s1a.reshape((self._feat_dim.past, self._feat_dim.fut_act, self._feat_dim.fut_obs))
        W_s1a = W_s1a.transpose((0,2,1))
        W_s1a = W_s1a.reshape((self._feat_dim.past,-1))                
        
        states = np.dot(feats.past, W_s1a)
        
        print('Stage 1B Regression')
        s1b_in = ula.khatri_rao_rowwise(feats.past, feats.exfut_act)
        s1b_out = feats.exfut_obs
        W_s1b = ridge(s1b_in, s1b_out, self._lambda['s1b'], imp_weights[1])
        #W_s1b = W_s1b.reshape((self._feat_dim.past, -1))
        W_s1b = W_s1b.reshape((self._feat_dim.past, self._feat_dim.exfut_act, self._feat_dim.exfut_obs))
        W_s1b = W_s1b.transpose((0,2,1))
        W_s1b = W_s1b.reshape((self._feat_dim.past,-1))                
        
        ex_states = np.dot(feats.past, W_s1b)
        
        print('Stage 1C Regression')
        s1c_in = ula.khatri_rao_rowwise(feats.past, feats.act)
        s1c_out = feats.oo
        W_s1c = ridge(s1c_in, s1c_out, self._lambda['s1c'], None)
        #W_s1c = W_s1c.reshape((self._feat_dim.past, -1))
        W_s1c = W_s1c.reshape((self._feat_dim.past, self._feat_dim.act, self._feat_dim.oo))
        W_s1c = W_s1c.transpose((0,2,1))
        W_s1c = W_s1c.reshape((self._feat_dim.past,-1))
        
        im_states = np.dot(feats.past, W_s1c)
        
        self._dbg_W_s1a = W_s1a
        self._dbg_W_s1b = W_s1b
        self._dbg_W_s1c = W_s1c                
        
        return states, ex_states, im_states
             
    def _build_model(self, data, feats, states): 
        self._state0 = np.mean(states,0)
        
        # Statistics for clamping
        self._max_state_norm2 = np.max(np.sum(states * states, 0))
        self._max_state_norm = np.sqrt(self._max_state_norm2)
        self._max_state_coord = np.max(states, axis=1).reshape((-1,1))
        self._min_state_coord = np.min(states, axis=1).reshape((-1,1))
        
        # Horizon Prediction        
        s2_h_in = ula.khatri_rao_rowwise(states, feats.fut_act)
        W_h = ridge(s2_h_in, feats.fut_obs, self._lambda['pred'])
        W_rff2fo = ridge(feats.fut_obs, data.fut_obs, self._lambda['pred'])
        self._W_h = np.dot(W_h, W_rff2fo)        
                                
        # 1-Step Prediction
        s2_1s_in = ula.khatri_rao_rowwise(states, feats.act)
        W_1s = ridge(s2_1s_in, feats.obs, self._lambda['pred'])
        W_rff2obs = ridge(feats.obs, data.obs, self._lambda['pred'])
        self._W_1s = np.dot(W_1s, W_rff2obs)        
        
    def _clamp_state_l2norm(self, state):        
        ss_norm2 = np.sum(state*state)
        if ss_norm2 > self._max_state_norm2:
            state *= (self._max_state_norm / np.sqrt(ss_norm2))
        return state
        
    def _clamp_state_coord(self, state):        
        return np.minimum(self._max_state_coord, np.maximum(self._min_state_coord, state))
                
      
    '''
    This is a section of experimental methods for discrete systems
    '''
        
    def _count_state(self, obs, act, past, importance_weights):
        if importance_weights is None:
            importance_weights = 1.0
        
        d_o = obs.shape[0]
        d_a = act.shape[0]
        d_h = past.shape[0]

        W = np.zeros((d_o*d_a,d_h))

        assert np.all(np.sum(obs,axis=0) == 1.0)
        assert np.all(np.sum(act,axis=0) == 1.0)
        assert np.all(np.sum(past,axis=0) == 1.0)
                        
        for i in xrange(d_o):
            for j in xrange(d_a):
                for k in xrange(d_h):
                    c_ijk = np.sum(obs[i,:] * act[j,:] * past[k,:] * importance_weights)
                    c_jk = np.sum(act[j,:] * past[k,:] * importance_weights)
                    
                    W[i+j*d_o,k] = (c_ijk + 1.0) / (c_jk + d_o) 
        
        idx = np.dot(np.array(xrange(d_h)), past)
        states = W[:,list(idx)]
                   
        return states
     
    def _diag_imstates(self, im_states_tmp):                
        d_oo = self._feat_dim.oo
        d_o = self._feat_dim.obs
        d_a = self._feat_dim.act
                
        n = im_states_tmp.shape[1]
        im_states = np.zeros((d_oo * d_a, n))
                                
        for a in xrange(d_a):
            idx_im=a*d_oo
            idx=a*d_o
            
            for o in xrange(d_o):
                im_states[idx_im,:] = im_states_tmp[idx,:]
                idx += 1
                idx_im += (d_o+1)
                
        return im_states
        
    def _s1_regression_count(self, data, feats, imp_weights):
        print('Stage 1A Regression')                
        states = self._count_state(feats.fut_obs, feats.fut_act, feats.past, imp_weights[0])
                
        print('Stage 1B Regression')
        ex_states = self._count_state(feats.exfut_obs, feats.exfut_act, feats.past, imp_weights[1])
        
        print('Stage 1C Regression')        
        im_states_tmp = self._count_state(feats.fut_obs, feats.fut_act, feats.past, None)
        im_states = self._diag_imstates(im_states_tmp)
                
        return states, ex_states, im_states

    def _dist2dist_states(self, obs, act, past, pv):
        d_o = obs.shape[0]
        d_a = act.shape[0]
        d_h = past.shape[0]

        counts = np.empty((d_o,d_a,d_h))        
        
        for i in xrange(d_o):
            for j in xrange(d_a):
                for k in xrange(d_h):
                    counts[i,j,k] = np.sum(obs[i,:] * act[j,:] * past[k,:])
                    
        def grad(W):
            W = W.reshape((-1,d_h),order='F')
            g = np.zeros_like(W)
            
            for i in xrange(d_o):
                for j in xrange(d_a):
                    for k in xrange(d_h):
                        pv_h = pv[:,k]
                        diff = pv_h * W[:,k]
                        diff[j*d_o+i] -= 1
                        
                        diff *= pv_h                                                                                                
                        g[:,k] += counts[i,j,k] * diff

            return g.ravel(order='F')                                     
            #return opt.numerical_jacobian(W,obj,1e-8).ravel()
            
        def obj(W):
            W = W.reshape((-1,d_h),order='F')
            val = 0.0
            
            for i in xrange(d_o):
                for j in xrange(d_a):
                    for k in xrange(d_h):
                        pv_h = pv[:,k]
                        diff = pv_h * W[:,k]
                        diff[j*d_o+i] -= 1
                                                
                        val += 0.5 * counts[i,j,k] * np.sum(diff*diff)

            return val
        
        def gd_callback(t, w, g, obj):
            if t % 100 == 0:
                print('Iteration = %d   GradNorm = %e  ' % (t,npla.norm(g)), end='')
                print('Error = %e' % obj)
         
        opt.validate_jacobian(d_o*d_a*d_h,obj,grad,1e-8)
                
        grad_fn = opt.AdamUpdater(d_o*d_a*d_h).create_grad_fn_wrapper(grad)        
        
        W = opt.grad_descent(d_o*d_a*d_h, grad_fn, obj_fn=obj, 
                             callback=gd_callback).reshape((-1,d_h),order='F')
                        
        states = np.dot(W,past)      
        return states
                        
    '''
    This is an experimental mode that is intended for discrete models.
    The property policy_vec_fn must be set.
    '''
    def _s1_regression_dist2dist(self, data, feats, imp_weights):
        pv_fut, pv_exfut, pv_im = self.policy_vec_fn(data, feats)
                        
        print('Stage 1A Regression')
        states = self._dist2dist_states(feats.fut_obs, feats.fut_act, feats.past, pv_fut)
        print('Stage 1B Regression')
        ex_states = self._dist2dist_states(feats.exfut_obs, feats.exfut_act, feats.past, pv_exfut)
        print('Stage 1C Regression')
        im_states_tmp = self._dist2dist_states(feats.fut_obs, feats.fut_act, feats.past, pv_im)
        im_states = self._diag_imstates(im_states_tmp)
        
        return states, ex_states, im_states
        
'''
Special feature extractors used to adapt RFFPSR to work in uncontrolled mode.
'''
class ConstFeatureExtractor(feat_extractor.FeatureExtractor):        
    def _process(self, raw_data):
        return np.ones((raw_data.shape[0],1))
        
'''
Wrapper for a feature extractor that makes it operate only on the first d
input coordinates. 
'''
class SubsetFeatureExtractor(feat_extractor.FeatureExtractor):
    def __init__(self, base_extractor, d):
        feat_extractor.FeatureExtractor.__init__(self)
        self._base = base_extractor
        self._d = d
    
    def _build(self, all_raw_data):
        self._base._build(all_raw_data[:,:self._d])
        
    def process(self, raw_data):
        return self._base.process(raw_data[:,:self._d])
  
class UncontrolledModelAdapter(UncontrolledModel):
    def __init__(self, controlled_model):
        self._controlled_model = controlled_model
        
    def train(self, traj_obs):
        traj_act = [np.ones((x.shape[0],1)) for x in traj_obs]
        return self._controlled_model.train(traj_obs, traj_act)
            
    def update_state(self, state, obs):
        return self._controlled_model.update_state(state, obs, np.array([1]))
        
    def predict_shifted_state(self, state):
        return self._controlled_model.predict_shifted_state(state, np.array([1]))
        
    def predict_obs(self, state):
        return self._controlled_model.predict_obs(state, np.array([1]))
        
    def predict_horizon(self, state):
        fut_act = np.ones((self.horizon_length,1))
        return self._controlled_model.predict_horizon(state, fut_act)
                        
    @property
    def state_dimension(self):
        return self._controlled_model.state_dimension        
        
    @property
    def horizon_length(self):
        return self._controlled_model.horizon_length        

    @property
    def initial_state(self):
        return self._controlled_model.initial_state        
    
        
class UncontrolledRFFPSR(UncontrolledModelAdapter):
    def __init__(self, obs_dim, future_length, past_length, projection_dim=50,
                l2_lambda=1e-3, feature_set = feat_extractor.create_RFFPCA_featureset(1000, 50)):
        feature_set = feature_set.copy()
        const_extractor = ConstFeatureExtractor()
        feature_set['act'] = const_extractor
        feature_set['fut_act'] = const_extractor
        feature_set['past'] = SubsetFeatureExtractor(feature_set['past'], obs_dim*past_length)
        
        rffpsr = RFFPSR(future_length, past_length, projection_dim, l2_lambda, feature_set, s1_method='joint', rng=np.random.RandomState(0))
        UncontrolledModelAdapter.__init__(self, rffpsr)
